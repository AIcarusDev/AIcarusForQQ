import asyncio
import json
import os
import struct
import sys
import unittest

import websockets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tts.server import TTSServer


PLUGIN_ID = "com.test.tts"


def _register_payload(secret_token: str = "secret") -> dict:
    return {
        "type": "register",
        "protocol_version": "1.0",
        "plugin_id": PLUGIN_ID,
        "secret_token": secret_token,
        "audio_format": {"sample_rate": 16000, "channels": 1, "bit_depth": 16},
        "llm_schema": {"properties": {"emotion": {"type": "string"}}},
    }


def _audio_frame(task_id: str, pcm: bytes) -> bytes:
    task_id_bytes = task_id.encode("utf-8")
    return struct.pack(">I", len(task_id_bytes)) + task_id_bytes + pcm


class TTSServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.audio_chunks: list[tuple[str, bytes]] = []
        self.server = TTSServer(
            host="127.0.0.1",
            port=0,
            secret_token="secret",
            on_audio_chunk=lambda task_id, pcm: self.audio_chunks.append((task_id, pcm)),
            ping_interval=0.05,
            pong_timeout=5.0,
        )
        await self.server.start()
        self.uri = f"ws://127.0.0.1:{self.server.bound_port}"

    async def asyncTearDown(self) -> None:
        await self.server.stop()

    async def _connect_registered_worker(self):
        ws = await websockets.connect(self.uri)
        await ws.send(json.dumps(_register_payload()))
        ack = json.loads(await ws.recv())
        self.assertTrue(ack.get("accepted"), ack)
        return ws

    async def test_register_dispatch_audio_and_complete(self) -> None:
        async with await self._connect_registered_worker() as ws:
            task_id = await self.server.dispatch_task(
                PLUGIN_ID,
                "hello",
                {"emotion": "neutral"},
            )
            task = json.loads(await ws.recv())
            self.assertEqual(task["type"], "task")
            self.assertEqual(task["task_id"], task_id)
            self.assertEqual(task["text"], "hello")
            self.assertEqual(task["parameters"], {"emotion": "neutral"})

            await ws.send(_audio_frame(task_id, b"pcm-data"))
            await ws.send(json.dumps({
                "type": "status",
                "task_id": task_id,
                "status": "completed",
            }))
            await self.server.wait_task(task_id, timeout=1.0)

        await asyncio.sleep(0.05)
        self.assertEqual(self.audio_chunks, [(task_id, b"pcm-data")])
        self.assertEqual(self.server.list_plugins(), [])

    async def test_rejects_invalid_token(self) -> None:
        async with websockets.connect(self.uri) as ws:
            await ws.send(json.dumps(_register_payload(secret_token="wrong")))
            ack = json.loads(await ws.recv())

        self.assertFalse(ack.get("accepted"))
        self.assertEqual(ack.get("reason"), "invalid_token")
        self.assertEqual(self.server.list_plugins(), [])

    async def test_limits_concurrent_tasks_per_plugin(self) -> None:
        await self.server.stop()
        self.server = TTSServer(
            host="127.0.0.1",
            port=0,
            secret_token="secret",
            max_concurrent_tasks_per_plugin=1,
        )
        await self.server.start()
        self.uri = f"ws://127.0.0.1:{self.server.bound_port}"

        async with await self._connect_registered_worker() as ws:
            first_task_id = await self.server.dispatch_task(PLUGIN_ID, "first")
            await ws.recv()
            with self.assertRaisesRegex(RuntimeError, "concurrent task limit"):
                await self.server.dispatch_task(PLUGIN_ID, "second")

            await ws.send(json.dumps({
                "type": "status",
                "task_id": first_task_id,
                "status": "completed",
            }))
            await self.server.wait_task(first_task_id, timeout=1.0)
            second_task_id = await self.server.dispatch_task(PLUGIN_ID, "second")
            second_task = json.loads(await ws.recv())
            self.assertEqual(second_task["task_id"], second_task_id)
            await ws.send(json.dumps({
                "type": "status",
                "task_id": second_task_id,
                "status": "completed",
            }))
            await self.server.wait_task(second_task_id, timeout=1.0)

    async def test_ignores_audio_for_unknown_task(self) -> None:
        async with await self._connect_registered_worker() as ws:
            await ws.send(_audio_frame("unknown", b"pcm-data"))
            await asyncio.sleep(0.05)

        self.assertEqual(self.audio_chunks, [])


if __name__ == "__main__":
    unittest.main()
