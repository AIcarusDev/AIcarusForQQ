# AIcarus for QQ

基于大语言模型的 QQ Bot，通过 QQ adapter 接入 QQ，支持私聊与群聊。

> 🌐 [English](README.md)

## 目录结构

- `src/`：核心源码
- `config/`：用户可编辑的配置与 Prompt 文档（`config_user.yaml`、`persona.md`、`style.md`、`social_tips/`、`self_image/`）
- `assets/`：只读静态资源（如 `voice_example.json`）
- `data/`：运行时持久化数据（SQLite 数据库、表情包收藏）
- `cache/`：运行时缓存，可安全删除（`image/`、`tts/`、`stickers/`）
- `logs/`：应用日志
- `templates/`：配置文件模板（`config.yaml.template`）

## 快速启动（Windows）

双击 **`start.bat`** 启动。

- **首次运行**：引导配置 Python 环境（创建 venv、使用 Conda 或系统 Python）并自动安装依赖。
- **后续运行**：记住上次选择，直接启动。

重置环境配置：

```cmd
start.bat --reset
```

## 配置

### 1. API Key（安全）

在项目根目录创建 `.env` 文件，填入对应供应商的 API Key：

```bash
# 例如
SILICONFLOW_API_KEY=sk-xxxxxxxx
```

**不要提交此文件。**

### 2. 常规设置

首次启动时，会自动从 `templates/config.yaml.template` 生成 `config/config_user.yaml`。可直接编辑该文件，或通过 WebUI 配置模型供应商、Bot 名称、QQ adapter 连接等。

该文件已被 git 忽略，个人配置不会入库。

### 3. QQ adapter

项目本身只监听 `qq_adapter.host` / `qq_adapter.port`，例如默认的 `ws://127.0.0.1:8078`。

在 NapCat 或 LLoneBot 里手动配置 OneBot v11 反向 WebSocket 连接到这个地址；同时开启上报自身消息，并使用 array 消息格式。项目不读取也不修改 adapter 的本地配置目录。

### 4. 人设

编辑 `config/persona.md` 设定 Bot 的性格与背景。

### 5. 自身形象

将 Bot 的形象图片（PNG/JPG/WEBP）放入 `config/self_image/`，启用视觉功能后 Bot 可通过 `get_self_image` 工具读取。该目录已被 git 忽略。

## 开发

不使用 `start.bat` 时手动启动：

```bash
# 激活你的 Python 环境并确保依赖已安装
python run.py
```

`run.py` 会将 `src/` 加入 `sys.path` 并启动应用。

## 许可证

本项目基于 [GNU Affero General Public License v3.0](LICENSE) 开源。
