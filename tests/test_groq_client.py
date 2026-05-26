import os
import unittest
from unittest.mock import patch

from agent.groq_client import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENAI_MODEL,
    get_llm_config,
    request_chat_completion,
)


class GroqClientConfigTests(unittest.TestCase):
    def test_default_config_uses_groq_and_recommended_model(self):
        with patch.dict(os.environ, {}, clear=True):
            config = get_llm_config(load_env_file=False)

        self.assertEqual(config["provider"], "groq")
        self.assertEqual(config["model"], DEFAULT_GROQ_MODEL)
        self.assertFalse(config["api_key_configured"])

    def test_groq_config_reads_api_key_and_model(self):
        with patch.dict(
            os.environ,
            {
                "GROQ_API_KEY": "test-groq-key",
                "KAIROS_LLM_MODEL": "llama-3.1-8b-instant",
                "KAIROS_LLM_PROVIDER": "groq",
            },
            clear=True,
        ):
            config = get_llm_config(load_env_file=False)

        self.assertEqual(config["provider"], "groq")
        self.assertEqual(config["model"], "llama-3.1-8b-instant")
        self.assertTrue(config["api_key_configured"])

    def test_groq_config_replaces_stale_openai_model_with_default(self):
        with patch.dict(
            os.environ,
            {
                "GROQ_API_KEY": "test-groq-key",
                "KAIROS_LLM_MODEL": "gpt-4o-mini",
                "KAIROS_LLM_PROVIDER": "groq",
            },
            clear=True,
        ):
            config = get_llm_config(load_env_file=False)

        self.assertEqual(config["provider"], "groq")
        self.assertEqual(config["model"], DEFAULT_GROQ_MODEL)
        self.assertIn("not a Groq model", config["warnings"][0])

    def test_openai_config_replaces_groq_model_with_openai_default(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai-key",
                "KAIROS_LLM_MODEL": "llama-3.3-70b-versatile",
                "KAIROS_LLM_PROVIDER": "openai",
            },
            clear=True,
        ):
            config = get_llm_config(load_env_file=False)

        self.assertEqual(config["provider"], "openai")
        self.assertEqual(config["model"], DEFAULT_OPENAI_MODEL)
        self.assertTrue(config["api_key_configured"])
        self.assertIn("not an OpenAI model", config["warnings"][0])

    def test_deterministic_config_has_no_api_key_warning(self):
        with patch.dict(
            os.environ,
            {"KAIROS_LLM_PROVIDER": "deterministic"},
            clear=True,
        ):
            config = get_llm_config(load_env_file=False)

        self.assertEqual(config["provider"], "deterministic")
        self.assertEqual(config["api_key_name"], "")
        self.assertFalse(config["api_key_configured"])
        self.assertEqual(config["warnings"], [])

    def test_missing_groq_api_key_has_clear_error(self):
        with patch.dict(os.environ, {"KAIROS_LLM_PROVIDER": "groq"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY is not set"):
                request_chat_completion(
                    [{"role": "user", "content": "Return JSON"}],
                    load_env_file=False,
                )


if __name__ == "__main__":
    unittest.main()
