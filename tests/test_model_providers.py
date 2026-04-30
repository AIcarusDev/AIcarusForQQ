import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm.core.profiles import get_model_providers, sanitize_model_providers


class ModelProviderNormalizationTests(unittest.TestCase):
    def test_get_model_providers_preserves_explicit_blank_name_and_api_key_env(self) -> None:
        providers = get_model_providers({
            "model_providers": {
                "custom": {
                    "name": "   ",
                    "base_url": " http://localhost:1234/v1 ",
                    "api_key_env": "",
                    "requires_api_key": False,
                    "supports_response_format": False,
                }
            }
        })

        self.assertEqual(providers["custom"]["name"], "")
        self.assertEqual(providers["custom"]["base_url"], "http://localhost:1234/v1")
        self.assertEqual(providers["custom"]["api_key_env"], "")
        self.assertFalse(providers["custom"]["requires_api_key"])
        self.assertFalse(providers["custom"]["supports_response_format"])

    def test_sanitize_model_providers_deduplicates_non_empty_display_names_only(self) -> None:
        providers = sanitize_model_providers(
            {
                "first": {"name": "LM Studio"},
                "second": {"name": "LM Studio"},
                "third": {"name": ""},
                "fourth": {"name": "   "},
                "fifth": {"name": "LM Studio"},
            },
            dedupe_display_names=True,
        )

        self.assertEqual(providers["first"]["name"], "LM Studio")
        self.assertEqual(providers["second"]["name"], "LM Studio(1)")
        self.assertEqual(providers["third"]["name"], "")
        self.assertEqual(providers["fourth"]["name"], "")
        self.assertEqual(providers["fifth"]["name"], "LM Studio(2)")

    def test_get_model_providers_uses_provider_id_when_name_is_missing(self) -> None:
        providers = get_model_providers({
            "model_providers": {
                "lmstudio": {"base_url": "http://localhost:1234/v1"}
            }
        })

        self.assertEqual(providers["lmstudio"]["name"], "lmstudio")


if __name__ == "__main__":
    unittest.main()