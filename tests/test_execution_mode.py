import os
import unittest

from execution_mode import ExecutionMode, get_execution_mode, resolve_execution_mode


class TestExecutionMode(unittest.TestCase):
    def test_default_mode_is_local(self):
        self.assertEqual(resolve_execution_mode(None), ExecutionMode.LOCAL)
        self.assertEqual(resolve_execution_mode(""), ExecutionMode.LOCAL)

    def test_accepts_remote_inference(self):
        self.assertEqual(resolve_execution_mode("REMOTE_INFERENCE"), ExecutionMode.REMOTE_INFERENCE)
        self.assertEqual(resolve_execution_mode("remote_inference"), ExecutionMode.REMOTE_INFERENCE)

    def test_invalid_value_falls_back_to_local(self):
        self.assertEqual(resolve_execution_mode("SOMETHING_ELSE"), ExecutionMode.LOCAL)

    def test_env_resolution(self):
        prev = os.environ.get("ANDESCODE_EXECUTION_MODE")
        try:
            os.environ["ANDESCODE_EXECUTION_MODE"] = "REMOTE_INFERENCE"
            self.assertEqual(get_execution_mode(), ExecutionMode.REMOTE_INFERENCE)
            os.environ["ANDESCODE_EXECUTION_MODE"] = "invalid"
            self.assertEqual(get_execution_mode(), ExecutionMode.LOCAL)
        finally:
            if prev is None:
                os.environ.pop("ANDESCODE_EXECUTION_MODE", None)
            else:
                os.environ["ANDESCODE_EXECUTION_MODE"] = prev
