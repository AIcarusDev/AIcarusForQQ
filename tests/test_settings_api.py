import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 在导入 main 之前 mock 掉 load_config，确保使用干净的测试配置
# 同时 mock create_adapter 避免真实的 API 客户端初始化
with (
    patch('config_loader.load_config') as mock_load_config,
    patch('llm.core.provider.create_adapter') as mock_create,
    patch('llm.media.vision_bridge.VisionBridge') as mock_vb,
):
    mock_load_config.return_value = (
        {
            "model": "fake-model",
            "provider": "custom",
            "model_providers": {
                "custom": {
                    "name": "Custom",
                    "base_url": "http://localhost/v1",
                    "api_key_env": "",
                }
            },
            "generation": {
                "enable_thinking": True,
            },
            "timezone": "Asia/Shanghai",
            "is": {
                "enabled": False,
            },
            "memory": {
                "auto_archive": {
                    "provider": "custom",
                    "model": "fake-model",
                }
            },
            "slow_thinking": {
                "enabled": False,
            },
            "vision_bridge": {
                "enabled": False,
            }
        },
        {"persona": "persona", "style": "style", "social_tips_private": "tips", "social_tips_group": "tips"}
    )
    from main import app


class SettingsApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_settings(self) -> None:
        client = app.test_client()
        response = await client.get('/settings/full')
        self.assertEqual(response.status_code, 200)
        data = await response.get_json()
        self.assertIn("generation", data)
        self.assertEqual(data["generation"].get("enable_thinking"), True)

    async def test_save_settings_enable_thinking(self) -> None:
        client = app.test_client()
        # 获取现有的配置作为基础，防止 schema 校验报错
        get_resp = await client.get('/settings/full')
        payload = await get_resp.get_json()
        
        # 改变 enable_thinking 的值
        payload["generation"]["enable_thinking"] = False
        
        # Mock save_config, env 写入函数, create_adapter 以及 VisionBridge 避免副作用
        with (
            patch('web.routes_settings.save_config') as mock_save_config,
            patch('web.routes_settings.save_env_key') as mock_save_env_key,
            patch('web.routes_settings.save_env_value') as mock_save_env_value,
            patch('web.routes_settings.save_env_proxy') as mock_save_env_proxy,
            patch('web.routes_settings.save_env_smtp') as mock_save_env_smtp,
            patch('web.routes_settings.save_env_imap') as mock_save_env_imap,
            patch('web.routes_settings.create_adapter') as mock_create_adapter,
            patch('web.routes_settings.VisionBridge') as mock_vision_bridge,
        ):
            resp = await client.post('/settings/full', json=payload)
            self.assertEqual(resp.status_code, 200)
            result = await resp.get_json()
            self.assertEqual(result.get("success"), True)
            
            # 确认 mock_save_config 被调用了，且传参包含我们修改后的值
            self.assertTrue(mock_save_config.called)
            saved_cfg = mock_save_config.call_args[0][0]
            self.assertEqual(saved_cfg["generation"].get("enable_thinking"), False)


if __name__ == "__main__":
    unittest.main()
