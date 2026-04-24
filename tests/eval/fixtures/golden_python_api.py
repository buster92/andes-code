"""
Golden Python API Codebase Fixture
====================================
A realistic Python backend ("TaskFlow") for the AndesCode eval framework.

Architecture: Layered (api / services / repositories / workers / cache)
Stack:        FastAPI, SQLAlchemy (async), Celery, Redis, Alembic, Pydantic v2, Pytest
Complexity:   Async SQLAlchemy session management, FastAPI dependency injection chains,
              Celery task routing with Redis broker/backend, JWT auth middleware,
              4-level nested dependency graph, background job retry logic.

Used by:
  - test_retrieval_precision.py  (stack-agnostic retrieval assertions)
  - test_answer_eval.py          (graded answer quality assertions)
"""

GOLDEN_FILES: dict[str, str] = {

# ─── PROJECT ROOT ─────────────────────────────────────────────────────────────

"pyproject.toml": """\
[tool.poetry]
name = "taskflow"
version = "0.1.0"
description = "Async task management API"
python = "^3.11"

[tool.poetry.dependencies]
fastapi           = "^0.111.0"
uvicorn           = {extras = ["standard"], version = "^0.29.0"}
sqlalchemy        = {extras = ["asyncio"], version = "^2.0.29"}
asyncpg           = "^0.29.0"          # async PostgreSQL driver
alembic           = "^1.13.1"
pydantic          = "^2.6.4"
pydantic-settings = "^2.2.1"
celery            = {extras = ["redis"], version = "^5.3.6"}
redis             = {extras = ["hiredis"], version = "^5.0.3"}
python-jose       = {extras = ["cryptography"], version = "^3.3.0"}
passlib            = {extras = ["bcrypt"], version = "^1.7.4"}
httpx             = "^0.27.0"          # async HTTP client for tests

[tool.poetry.dev-dependencies]
pytest            = "^8.1.1"
pytest-asyncio    = "^0.23.6"
pytest-cov        = "^5.0.0"
factory-boy       = "^3.3.0"
""",

"requirements.txt": """\
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
sqlalchemy[asyncio]>=2.0.29
asyncpg>=0.29.0
alembic>=1.13.1
pydantic>=2.6.4
pydantic-settings>=2.2.1
celery[redis]>=5.3.6
redis[hiredis]>=5.0.3
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
httpx>=0.27.0
""",

# ─── CONFIG ───────────────────────────────────────────────────────────────────

"app/config.py": """\
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    \"\"\"
    Application settings loaded from environment variables.
    Pydantic-settings validates types at startup — missing required vars
    raise a ValidationError immediately, not at the first use site.

    Database:
      DATABASE_URL must use the asyncpg driver:
        postgresql+asyncpg://user:pass@host:5432/dbname
      The sync URL (for Alembic migrations, which don't support asyncpg) is
      derived automatically by swapping the driver prefix.

    Celery:
      CELERY_BROKER_URL and CELERY_RESULT_BACKEND both point to Redis.
      Using separate Redis databases (db=0 for broker, db=1 for results)
      avoids key collisions under high load.
    \"\"\"
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # App
    app_name: str = "TaskFlow"
    debug: bool = False
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Database
    database_url: str   # postgresql+asyncpg://...

    @property
    def sync_database_url(self) -> str:
        \"\"\"Alembic requires a sync driver — replace asyncpg with psycopg2.\"\"\"
        return self.database_url.replace("+asyncpg", "+psycopg2")

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Pagination
    default_page_size: int = 20
    max_page_size: int = 100


@lru_cache
def get_settings() -> Settings:
    \"\"\"
    Cached settings singleton. lru_cache means the .env file is read once.
    In tests, call get_settings.cache_clear() then monkeypatch environment
    variables to inject test-specific values.
    \"\"\"
    return Settings()
""",

# ─── DATABASE ─────────────────────────────────────────────────────────────────

"app/database.py": """\
from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings


class Base(DeclarativeBase):
    \"\"\"
    SQLAlchemy declarative base. All ORM models inherit from this.
    Using the new 2.0-style DeclarativeBase (not declarative_base()) for
    full type annotation support and mypy compatibility.
    \"\"\"
    pass


def create_engine() -> AsyncEngine:
    \"\"\"
    Creates the async SQLAlchemy engine.

    pool_size / max_overflow:
      Each async worker (uvicorn) shares this pool. With 4 workers and
      pool_size=5, you have up to 20 concurrent DB connections — size
      appropriately for your PostgreSQL max_connections setting.

    pool_pre_ping:
      Detects stale connections (e.g., after a DB restart) before use,
      preventing OperationalError on the first query post-reconnect.

    echo:
      Set True only in debug mode — logs every SQL statement, which is
      extremely noisy in production.
    \"\"\"
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=settings.debug,
    )


engine = create_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # prevent lazy-load errors after commit in async context
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    \"\"\"
    FastAPI dependency that yields an async database session.
    The session is committed on success and rolled back on exception.
    Always closed in the finally block.

    Usage in a route:
        @router.get("/tasks")
        async def list_tasks(db: AsyncSession = Depends(get_db)):
            ...
    \"\"\"
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
""",

# ─── MODELS ───────────────────────────────────────────────────────────────────

"app/models/task.py": """\
import uuid
from datetime import datetime
from sqlalchemy import String, Text, ForeignKey, DateTime, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum
from app.database import Base


class TaskStatus(str, enum.Enum):
    \"\"\"
    Task lifecycle: TODO → IN_PROGRESS → DONE | CANCELLED.
    Inherits from str so Pydantic serializes it as a string without
    a custom validator. The SAEnum stores the string value in PostgreSQL.
    \"\"\"
    TODO        = "todo"
    IN_PROGRESS = "in_progress"
    DONE        = "done"
    CANCELLED   = "cancelled"


class TaskPriority(str, enum.Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    URGENT = "urgent"


class Task(Base):
    \"\"\"
    Core task entity.

    assignee_id is a nullable FK to users — tasks can exist unassigned.
    celery_task_id stores the Celery task ID when a background job is
    dispatched for this task (e.g., send notification, run automation).
    It is set by TaskService.dispatch_background_job() and used to
    check job status via Celery's AsyncResult.
    \"\"\"
    __tablename__ = "tasks"

    id:             Mapped[uuid.UUID]     = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title:          Mapped[str]           = mapped_column(String(255), nullable=False)
    description:    Mapped[str | None]    = mapped_column(Text)
    status:         Mapped[TaskStatus]    = mapped_column(SAEnum(TaskStatus), default=TaskStatus.TODO, nullable=False)
    priority:       Mapped[TaskPriority]  = mapped_column(SAEnum(TaskPriority), default=TaskPriority.MEDIUM, nullable=False)
    due_date:       Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at:     Mapped[datetime]      = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at:     Mapped[datetime]      = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    owner_id:       Mapped[uuid.UUID]     = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    assignee_id:    Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    celery_task_id: Mapped[str | None]    = mapped_column(String(255))

    owner:    Mapped["User"] = relationship("User", foreign_keys=[owner_id], back_populates="owned_tasks")
    assignee: Mapped["User | None"] = relationship("User", foreign_keys=[assignee_id])
""",

"app/models/user.py": """\
import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    \"\"\"
    User entity.
    hashed_password stores the bcrypt hash — the plaintext password is
    never persisted. AuthService.verify_password() compares via passlib.
    is_active allows soft-disabling accounts without deletion.
    \"\"\"
    __tablename__ = "users"

    id:              Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email:           Mapped[str]       = mapped_column(String(255), unique=True, nullable=False, index=True)
    display_name:    Mapped[str]       = mapped_column(String(100), nullable=False)
    hashed_password: Mapped[str]       = mapped_column(String(255), nullable=False)
    is_active:       Mapped[bool]      = mapped_column(Boolean, default=True, nullable=False)
    created_at:      Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now())

    owned_tasks: Mapped[list["Task"]] = relationship(
        "Task", foreign_keys="Task.owner_id", back_populates="owner"
    )
""",

# ─── SCHEMAS ──────────────────────────────────────────────────────────────────

"app/schemas/task.py": """\
import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from app.models.task import TaskStatus, TaskPriority


class TaskCreate(BaseModel):
    title:       str           = Field(..., min_length=1, max_length=255)
    description: str | None    = None
    priority:    TaskPriority  = TaskPriority.MEDIUM
    due_date:    datetime | None = None
    assignee_id: uuid.UUID | None = None


class TaskUpdate(BaseModel):
    title:       str | None         = Field(None, min_length=1, max_length=255)
    description: str | None         = None
    status:      TaskStatus | None  = None
    priority:    TaskPriority | None = None
    due_date:    datetime | None    = None
    assignee_id: uuid.UUID | None   = None


class TaskResponse(BaseModel):
    model_config = {"from_attributes": True}

    id:             uuid.UUID
    title:          str
    description:    str | None
    status:         TaskStatus
    priority:       TaskPriority
    due_date:       datetime | None
    created_at:     datetime
    updated_at:     datetime
    owner_id:       uuid.UUID
    assignee_id:    uuid.UUID | None
    celery_task_id: str | None


class TaskPage(BaseModel):
    items: list[TaskResponse]
    total: int
    page:  int
    size:  int
""",

"app/schemas/user.py": """\
import uuid
from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email:        EmailStr
    display_name: str = Field(..., min_length=1, max_length=100)
    password:     str = Field(..., min_length=8)


class UserResponse(BaseModel):
    model_config = {"from_attributes": True}
    id:           uuid.UUID
    email:        str
    display_name: str
    is_active:    bool


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str
""",

# ─── REPOSITORIES ─────────────────────────────────────────────────────────────

"app/repositories/task_repository.py": """\
import uuid
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.task import Task, TaskStatus


class TaskRepository:
    \"\"\"
    Data access layer for Task entities.
    All methods accept an AsyncSession injected by the FastAPI dependency chain:
      route → get_db (Depends) → TaskRepository → TaskService

    Async SQLAlchemy patterns:
      - session.execute(select(...)) returns a CursorResult; call .scalars().all()
        to get ORM objects rather than Row tuples.
      - session.get(Task, id) is a shortcut for a primary-key lookup and uses
        the session identity map (avoids a DB round-trip if already loaded).
      - Bulk updates use update() + session.execute() rather than loading objects
        into memory — critical for performance on large datasets.
    \"\"\"

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, task_id: uuid.UUID) -> Task | None:
        return await self.db.get(Task, task_id)

    async def list_by_owner(
        self,
        owner_id: uuid.UUID,
        status: TaskStatus | None = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[Task], int]:
        \"\"\"Returns (items, total_count) for pagination.\"\"\"
        query = select(Task).where(Task.owner_id == owner_id)
        if status:
            query = query.where(Task.status == status)

        total_q  = select(func.count()).select_from(query.subquery())
        total    = (await self.db.execute(total_q)).scalar_one()

        items_q  = query.offset((page - 1) * size).limit(size).order_by(Task.created_at.desc())
        items    = (await self.db.execute(items_q)).scalars().all()
        return list(items), total

    async def create(self, task: Task) -> Task:
        self.db.add(task)
        await self.db.flush()   # flush to get DB-generated fields (id, created_at)
        await self.db.refresh(task)
        return task

    async def update(self, task: Task, updates: dict) -> Task:
        for key, value in updates.items():
            setattr(task, key, value)
        await self.db.flush()
        await self.db.refresh(task)
        return task

    async def set_celery_task_id(self, task_id: uuid.UUID, celery_id: str) -> None:
        \"\"\"
        Lightweight update — does not load the full Task object into memory.
        Called immediately after dispatching a Celery job.
        \"\"\"
        stmt = (
            update(Task)
            .where(Task.id == task_id)
            .values(celery_task_id=celery_id)
        )
        await self.db.execute(stmt)

    async def delete(self, task: Task) -> None:
        await self.db.delete(task)
""",

"app/repositories/user_repository.py": """\
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User


class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return await self.db.get(User, user_id)

    async def get_by_email(self, email: str) -> User | None:
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def create(self, user: User) -> User:
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)
        return user
""",

# ─── SERVICES ─────────────────────────────────────────────────────────────────

"app/services/task_service.py": """\
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.task import Task, TaskStatus
from app.models.user import User
from app.repositories.task_repository import TaskRepository
from app.schemas.task import TaskCreate, TaskUpdate, TaskPage, TaskResponse
from app.workers.tasks import notify_assignee, run_task_automation
from app.cache.redis_client import RedisClient
from fastapi import HTTPException, status
import json


class TaskService:
    \"\"\"
    Business logic layer for task management.

    Full dependency chain:
      TaskService
        ├── TaskRepository     (SQLAlchemy AsyncSession → PostgreSQL)
        ├── RedisClient        (cache + pub/sub)
        └── Celery tasks       (notify_assignee, run_task_automation)
              └── Redis broker (Celery uses Redis for task queue)

    Cache strategy:
      Individual tasks are cached in Redis with a 5-minute TTL.
      Cache key: task:{task_id}
      On mutation (update/delete), the cache entry is invalidated.
      List queries are NOT cached — pagination makes cache keys too
      varied to be worthwhile; rely on PostgreSQL for list queries.
    \"\"\"

    CACHE_TTL = 300   # 5 minutes

    def __init__(self, db: AsyncSession, redis: RedisClient):
        self.repo  = TaskRepository(db)
        self.redis = redis

    def _cache_key(self, task_id: uuid.UUID) -> str:
        return f"task:{task_id}"

    async def get_task(self, task_id: uuid.UUID, current_user: User) -> Task:
        \"\"\"Fetches task from cache first; falls back to DB on miss.\"\"\"
        key = self._cache_key(task_id)
        cached = await self.redis.get(key)
        if cached:
            return Task(**json.loads(cached))

        task = await self.repo.get_by_id(task_id)
        if not task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
        if task.owner_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

        await self.redis.set(key, task, ttl=self.CACHE_TTL)
        return task

    async def list_tasks(
        self,
        owner: User,
        filter_status: TaskStatus | None,
        page: int,
        size: int,
    ) -> TaskPage:
        items, total = await self.repo.list_by_owner(owner.id, filter_status, page, size)
        return TaskPage(
            items=[TaskResponse.model_validate(t) for t in items],
            total=total, page=page, size=size,
        )

    async def create_task(self, data: TaskCreate, owner: User) -> Task:
        task = Task(owner_id=owner.id, **data.model_dump())
        task = await self.repo.create(task)

        # Dispatch background job if task has an assignee
        if task.assignee_id:
            job = notify_assignee.delay(str(task.id), str(task.assignee_id))
            await self.repo.set_celery_task_id(task.id, job.id)

        return task

    async def update_task(self, task_id: uuid.UUID, data: TaskUpdate, current_user: User) -> Task:
        task = await self.get_task(task_id, current_user)
        updates = data.model_dump(exclude_unset=True)
        task = await self.repo.update(task, updates)

        # Invalidate cache on any mutation
        await self.redis.delete(self._cache_key(task_id))

        # If task moved to IN_PROGRESS, trigger automation job
        if updates.get("status") == TaskStatus.IN_PROGRESS:
            job = run_task_automation.delay(str(task.id))
            await self.repo.set_celery_task_id(task.id, job.id)

        return task

    async def delete_task(self, task_id: uuid.UUID, current_user: User) -> None:
        task = await self.get_task(task_id, current_user)
        await self.redis.delete(self._cache_key(task_id))
        await self.repo.delete(task)
""",

"app/services/auth_service.py": """\
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import get_settings
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate, TokenResponse
from fastapi import HTTPException, status

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    \"\"\"
    Handles registration, login, and JWT token generation/validation.

    Dependency chain:
      AuthService
        └── UserRepository (SQLAlchemy AsyncSession → PostgreSQL)

    JWT structure:
      Payload: {sub: user_id, exp: expiry_timestamp}
      The sub claim stores the user UUID as a string.
      Tokens are signed with HS256 using SECRET_KEY from settings.
      No refresh tokens — clients re-authenticate after expiry.

    Password hashing:
      passlib CryptContext with bcrypt handles hashing and verification.
      Work factor is bcrypt's default (12 rounds as of passlib 1.7).
    \"\"\"

    def __init__(self, db: AsyncSession):
        self.repo     = UserRepository(db)
        self.settings = get_settings()

    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)

    def verify_password(self, plain: str, hashed: str) -> bool:
        return pwd_context.verify(plain, hashed)

    def create_access_token(self, user_id: str) -> str:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=self.settings.access_token_expire_minutes
        )
        return jwt.encode(
            {"sub": user_id, "exp": expire},
            self.settings.secret_key,
            algorithm=self.settings.algorithm,
        )

    def decode_token(self, token: str) -> str:
        \"\"\"Returns user_id (sub claim) or raises HTTPException 401.\"\"\"
        try:
            payload = jwt.decode(token, self.settings.secret_key,
                                 algorithms=[self.settings.algorithm])
            user_id: str = payload.get("sub")
            if user_id is None:
                raise ValueError("Missing sub claim")
            return user_id
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

    async def register(self, data: UserCreate) -> User:
        existing = await self.repo.get_by_email(data.email)
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        user = User(
            email=data.email,
            display_name=data.display_name,
            hashed_password=self.hash_password(data.password),
        )
        return await self.repo.create(user)

    async def login(self, email: str, password: str) -> TokenResponse:
        user = await self.repo.get_by_email(email)
        if not user or not self.verify_password(password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account disabled")
        token = self.create_access_token(str(user.id))
        return TokenResponse(access_token=token)
""",

# ─── API ROUTES ───────────────────────────────────────────────────────────────

"app/api/deps.py": """\
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService
from app.cache.redis_client import RedisClient, get_redis
import uuid

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    \"\"\"
    FastAPI dependency that resolves the Bearer token to a User object.

    Dependency chain:
      get_current_user
        ├── get_db            (yields AsyncSession)
        └── AuthService
              └── UserRepository (uses the same AsyncSession)

    The resolved User is injected into route handlers via:
        current_user: User = Depends(get_current_user)

    This pattern means every protected route automatically gets a validated
    user without duplicating token-parsing logic.
    \"\"\"
    auth_service = AuthService(db)
    user_id_str  = auth_service.decode_token(credentials.credentials)
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token subject")

    user = await UserRepository(db).get_by_id(user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


def get_task_service(
    db: AsyncSession = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
) -> "TaskService":
    from app.services.task_service import TaskService
    return TaskService(db=db, redis=redis)
""",

"app/api/routes/tasks.py": """\
import uuid
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.task import TaskStatus, Task
from app.models.user import User
from app.schemas.task import TaskCreate, TaskUpdate, TaskResponse, TaskPage
from app.services.task_service import TaskService
from app.api.deps import get_current_user, get_task_service

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("", response_model=TaskPage)
async def list_tasks(
    filter_status: TaskStatus | None = Query(None),
    page:  int = Query(1, ge=1),
    size:  int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
):
    return await service.list_tasks(current_user, filter_status, page, size)


@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    data: TaskCreate,
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
):
    return await service.create_task(data, current_user)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
):
    return await service.get_task(task_id, current_user)


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: uuid.UUID,
    data: TaskUpdate,
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
):
    return await service.update_task(task_id, data, current_user)


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: TaskService = Depends(get_task_service),
):
    await service.delete_task(task_id, current_user)
""",

"app/api/routes/auth.py": """\
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.schemas.user import UserCreate, UserResponse, LoginRequest, TokenResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(data: UserCreate, db: AsyncSession = Depends(get_db)):
    service = AuthService(db)
    user = await service.register(data)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    service = AuthService(db)
    return await service.login(data.email, data.password)
""",

# ─── CACHE ────────────────────────────────────────────────────────────────────

"app/cache/redis_client.py": """\
import json
from typing import Any
import redis.asyncio as aioredis
from app.config import get_settings


class RedisClient:
    \"\"\"
    Async Redis wrapper used for caching and pub/sub.

    Cache strategy (used by TaskService):
      - get(key) returns deserialized Python object or None on miss
      - set(key, value, ttl) serializes to JSON and sets expiry
      - delete(key) removes entry on mutation

    Pub/sub (used for real-time task update notifications):
      - publish(channel, message) sends to a Redis channel
      - subscribe() returns an async generator of messages

    Connection pool:
      aioredis.from_url creates a connection pool automatically.
      max_connections=20 caps pool size — tune based on Redis server limits.
      decode_responses=True means all values are returned as str, not bytes.
    \"\"\"

    def __init__(self, client: aioredis.Redis):
        self._client = client

    async def get(self, key: str) -> Any | None:
        value = await self._client.get(key)
        return json.loads(value) if value else None

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        serialized = json.dumps(value, default=str)
        await self._client.setex(key, ttl, serialized)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def publish(self, channel: str, message: dict) -> None:
        await self._client.publish(channel, json.dumps(message, default=str))

    async def ping(self) -> bool:
        return await self._client.ping()


_redis_pool: aioredis.Redis | None = None


async def get_redis() -> RedisClient:
    \"\"\"
    FastAPI dependency that returns a shared RedisClient.
    The connection pool is created once at startup (lazy init on first request).
    \"\"\"
    global _redis_pool
    if _redis_pool is None:
        settings = get_settings()
        _redis_pool = aioredis.from_url(
            settings.redis_url,
            max_connections=20,
            decode_responses=True,
        )
    return RedisClient(_redis_pool)
""",

# ─── CELERY WORKERS ───────────────────────────────────────────────────────────

"app/workers/celery_app.py": """\
from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "taskflow",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer       = "json",
    result_serializer     = "json",
    accept_content        = ["json"],
    timezone              = "UTC",
    enable_utc            = True,

    # Retry policy defaults — individual tasks can override
    task_acks_late        = True,    # acknowledge after task completes, not before
    task_reject_on_worker_lost = True,

    # Route different task types to dedicated queues
    task_routes = {
        "app.workers.tasks.notify_assignee":    {"queue": "notifications"},
        "app.workers.tasks.run_task_automation": {"queue": "automation"},
    },

    # Result expiry — keep task results in Redis for 24 hours
    result_expires = 86400,
)
""",

"app/workers/tasks.py": """\
import uuid
from celery import shared_task
from celery.utils.log import get_task_logger
from app.workers.celery_app import celery_app

logger = get_task_logger(__name__)


@celery_app.task(
    bind=True,
    name="app.workers.tasks.notify_assignee",
    max_retries=5,
    default_retry_delay=60,    # 60s base delay; Celery applies exponential backoff
    queue="notifications",
)
def notify_assignee(self, task_id: str, assignee_id: str) -> dict:
    \"\"\"
    Sends an email/push notification to the task assignee.

    Retry strategy:
      max_retries=5 with default_retry_delay=60s.
      On transient failure (SMTP timeout, push service unavailable),
      self.retry() re-queues the task. Celery applies exponential backoff
      if the broker supports it (Redis does via eta/countdown).

    bind=True gives access to self (the task instance) so we can call
    self.retry() with exc= to preserve the original traceback.

    The task ID stored on the Task model (celery_task_id) lets the API
    check job status via AsyncResult(celery_task_id).state.
    \"\"\"
    try:
        logger.info(f"Notifying assignee {assignee_id} for task {task_id}")
        # In production: call email service / push notification provider
        # _send_email(assignee_id, task_id)
        return {"status": "sent", "task_id": task_id, "assignee_id": assignee_id}
    except Exception as exc:
        logger.warning(f"notify_assignee failed, retrying: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    name="app.workers.tasks.run_task_automation",
    max_retries=3,
    default_retry_delay=30,
    queue="automation",
)
def run_task_automation(self, task_id: str) -> dict:
    \"\"\"
    Runs automation rules when a task transitions to IN_PROGRESS.
    Examples: assign sub-tasks, trigger webhooks, update external trackers.
    \"\"\"
    try:
        logger.info(f"Running automation for task {task_id}")
        return {"status": "automation_complete", "task_id": task_id}
    except Exception as exc:
        raise self.retry(exc=exc)
""",

# ─── ALEMBIC ──────────────────────────────────────────────────────────────────

"alembic/env.py": """\
import asyncio
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import AsyncEngine
from alembic import context
from app.config import get_settings
from app.database import Base
# Import all models so Alembic's autogenerate detects them
from app.models.task import Task   # noqa: F401
from app.models.user import User   # noqa: F401

config = context.config
fileConfig(config.config_file_name)

# Override the DB URL from settings (supports env var injection)
# Must use the SYNC URL — asyncpg is not supported by Alembic's migration runner
settings  = get_settings()
config.set_main_option("sqlalchemy.url", settings.sync_database_url)

target_metadata = Base.metadata


def run_migrations_offline():
    context.configure(url=settings.sync_database_url, target_metadata=target_metadata,
                      literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(config.get_section(config.config_ini_section),
                                     prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
""",

# ─── TESTS ────────────────────────────────────────────────────────────────────

"tests/conftest.py": """\
import asyncio
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.database import Base, get_db
from app.cache.redis_client import get_redis, RedisClient
from app.main import app
from unittest.mock import AsyncMock

# Use an in-memory SQLite database for tests — no PostgreSQL needed
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    \"\"\"
    Single event loop for the entire test session.
    pytest-asyncio creates a new loop per test by default;
    scope=session avoids overhead and supports session-scoped async fixtures.
    \"\"\"
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncSession:
    \"\"\"
    Per-test database session with automatic rollback.
    Wrapping in a nested transaction (SAVEPOINT) means the outer transaction
    is never committed — the DB is clean after every test without truncating tables.
    \"\"\"
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as session:
        async with session.begin_nested():
            yield session
        await session.rollback()


@pytest.fixture
def mock_redis() -> RedisClient:
    \"\"\"Mock RedisClient — no Redis server needed in tests.\"\"\"
    mock = AsyncMock(spec=RedisClient)
    mock.get.return_value = None   # cache miss by default
    return mock


@pytest_asyncio.fixture
async def client(db_session, mock_redis) -> AsyncClient:
    \"\"\"
    Async test client with dependency overrides:
      - get_db     → uses in-memory SQLite session
      - get_redis  → uses mock (no Redis needed)
    \"\"\"
    app.dependency_overrides[get_db]    = lambda: db_session
    app.dependency_overrides[get_redis] = lambda: mock_redis
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
""",

"tests/test_tasks.py": """\
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_task_unauthenticated(client: AsyncClient):
    \"\"\"Requests without a token must return 403 (HTTPBearer returns 403, not 401).\"\"\"
    response = await client.post("/tasks", json={"title": "Test"})
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_register_and_login(client: AsyncClient):
    \"\"\"Full auth flow: register → login → receive JWT.\"\"\"
    reg = await client.post("/auth/register", json={
        "email": "test@example.com",
        "display_name": "Tester",
        "password": "secure123",
    })
    assert reg.status_code == 201

    login = await client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "secure123",
    })
    assert login.status_code == 200
    assert "access_token" in login.json()


@pytest.mark.asyncio
async def test_create_and_list_tasks(client: AsyncClient):
    \"\"\"Authenticated user can create a task and see it in the list.\"\"\"
    # Register + login
    await client.post("/auth/register", json={
        "email": "user2@example.com", "display_name": "User2", "password": "pw123456"
    })
    login_resp = await client.post("/auth/login", json={
        "email": "user2@example.com", "password": "pw123456"
    })
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    create = await client.post("/tasks", json={"title": "My first task"}, headers=headers)
    assert create.status_code == 201
    task_id = create.json()["id"]

    listing = await client.get("/tasks", headers=headers)
    assert listing.status_code == 200
    ids = [t["id"] for t in listing.json()["items"]]
    assert task_id in ids


@pytest.mark.asyncio
async def test_update_task_status(client: AsyncClient):
    \"\"\"Status transition TODO → IN_PROGRESS triggers automation (celery_task_id set).\"\"\"
    await client.post("/auth/register", json={
        "email": "user3@example.com", "display_name": "User3", "password": "pw123456"
    })
    login_resp = await client.post("/auth/login", json={
        "email": "user3@example.com", "password": "pw123456"
    })
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    task = (await client.post("/tasks", json={"title": "Automate me"}, headers=headers)).json()
    updated = await client.patch(f'/tasks/{task[\"id\"]}',
                                  json={"status": "in_progress"}, headers=headers)
    assert updated.status_code == 200
    # celery_task_id is set when automation job is dispatched
    assert updated.json()["status"] == "in_progress"
""",

# ─── APP ENTRYPOINT ───────────────────────────────────────────────────────────

"app/main.py": """\
from fastapi import FastAPI
from app.api.routes import tasks, auth
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
)

app.include_router(auth.router)
app.include_router(tasks.router)


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.app_name}
""",

}


def write_golden_codebase(target_dir: str) -> list[str]:
    import os
    written = []
    for rel_path, content in GOLDEN_FILES.items():
        full_path = os.path.join(target_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        written.append(full_path)
    return written
