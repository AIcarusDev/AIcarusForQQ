import os
import sys
import unittest
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tools.send_voice_message import (
    DECLARATION,
    _wav_duration_seconds,
    _write_pcm_wav,
    condition,
    sanitize_semantic_args,
)


class SendVoiceMessageToolTests(unittest.TestCase):
    def test_condition_requires_tts_enabled(self) -> None:
        self.assertFalse(condition({}))
        self.assertFalse(condition({"tts": {"enabled": False}}))
        self.assertTrue(condition({"tts": {"enabled": True}}))

    def test_schema_only_accepts_single_text(self) -> None:
        params = DECLARATION["parameters"]
        self.assertEqual(params["required"], ["motivation", "text"])
        self.assertIn("text", params["properties"])
        self.assertNotIn("messages", params["properties"])

    def test_sanitize_trims_text(self) -> None:
        args, changes, error = sanitize_semantic_args({"text": "  你好  "})
        self.assertIsNone(error)
        self.assertEqual(args["text"], "你好")
        self.assertEqual(changes, ["trimmed text"])

    def test_write_pcm_wav_uses_audio_format(self) -> None:
        path = _write_pcm_wav(
            b"\x00\x00\x01\x00",
            {"sample_rate": 16000, "channels": 1, "bit_depth": 16},
        )
        try:
            with wave.open(str(path), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 16000)
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.readframes(2), b"\x00\x00\x01\x00")
            self.assertAlmostEqual(_wav_duration_seconds(path), 2 / 16000)
        finally:
            path.unlink(missing_ok=True)

    def test_voice_segment_renders_with_duration(self) -> None:
        from llm.prompt.xml_builder import _render_content_chunks

        self.assertEqual(
            _render_content_chunks([{"type": "voice", "duration": 1.2}]),
            [("voice", "[语音 1'']")],
        )
        self.assertEqual(
            _render_content_chunks([{"type": "voice", "duration": 83}]),
            [("voice", "[语音 1'23'']")],
        )


if __name__ == "__main__":
    unittest.main()