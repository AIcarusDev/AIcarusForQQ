import unittest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qq_adapter.client import QQAdapterClient


class QQAdapterClientConnectionTests(unittest.TestCase):
    def test_stale_handler_cleanup_does_not_clear_new_connection(self) -> None:
        client = QQAdapterClient()
        old_ws = object()
        new_ws = object()

        client._ws = new_ws  # type: ignore[assignment]
        client.bot_id = "213628848"
        client._last_heartbeat_at = 123.0

        cleared = client._clear_connection_if_current(old_ws)  # type: ignore[arg-type]

        self.assertFalse(cleared)
        self.assertIs(client._ws, new_ws)
        self.assertEqual(client.bot_id, "213628848")
        self.assertEqual(client._last_heartbeat_at, 123.0)

    def test_current_handler_cleanup_clears_connection(self) -> None:
        client = QQAdapterClient()
        ws = object()

        client._ws = ws  # type: ignore[assignment]
        client.bot_id = "213628848"
        client._last_heartbeat_at = 123.0
        client._conv_locks["group_1"] = object()  # type: ignore[assignment]
        client._ready.set()

        cleared = client._clear_connection_if_current(ws)  # type: ignore[arg-type]

        self.assertTrue(cleared)
        self.assertIsNone(client._ws)
        self.assertIsNone(client.bot_id)
        self.assertEqual(client._last_heartbeat_at, 0.0)
        self.assertFalse(client._conv_locks)
        self.assertFalse(client._ready.is_set())
