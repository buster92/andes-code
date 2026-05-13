from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class RetrievalProvider(Protocol):
    def build_context(
        self,
        messages: list[dict[str, Any]],
        request_id: str,
        *,
        debug_mode: bool = False,
        return_debug: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any] | None]:
        ...


class AnswerContextBuilder(Protocol):
    def build_prompt(self, messages: list[dict[str, Any]]) -> str:
        ...


class AnswerEngine(Protocol):
    def generate(self, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        ...


@dataclass
class LocalAskOrchestrator:
    retrieval: RetrievalProvider
    context_builder: AnswerContextBuilder
    answer_engine: AnswerEngine
    strip_thinking: Any
    is_performance_query: Any
    validate_high_signal_output: Any

    def run_non_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        request_id: str,
        max_tokens: int,
        debug_mode: bool,
    ) -> tuple[str, dict[str, Any] | None]:
        contextualized, debug_payload = self.retrieval.build_context(
            messages,
            request_id,
            debug_mode=debug_mode,
            return_debug=True,
        )
        user_query = next(
            (m.get("content", "") for m in reversed(contextualized) if m.get("role") == "user"),
            "",
        )
        is_performance = self.is_performance_query(user_query)
        prompt = self.context_builder.build_prompt(contextualized)
        result = self.answer_engine.generate(prompt, max_tokens=max_tokens)
        text = self.strip_thinking(result["choices"][0]["text"], strip_edges=True)
        filtered_text, _ = self.validate_high_signal_output(text, is_performance)
        return filtered_text, debug_payload


@dataclass
class FunctionRetrievalProvider:
    build_context_fn: Any

    def build_context(
        self,
        messages: list[dict[str, Any]],
        request_id: str,
        *,
        debug_mode: bool = False,
        return_debug: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any] | None]:
        return self.build_context_fn(messages, request_id, debug_mode=debug_mode, return_debug=return_debug)


@dataclass
class FunctionAnswerContextBuilder:
    to_prompt_fn: Any

    def build_prompt(self, messages: list[dict[str, Any]]) -> str:
        return self.to_prompt_fn(messages)


@dataclass
class LlamaAnswerEngine:
    llm: Any

    def generate(self, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        return self.llm(prompt, max_tokens=max_tokens, echo=False)
