# AIcarusForQQ

QQ AI Bot with LLM integration (Restructured).

> 🌐 [中文文档](README.zh.md)

## Structure

- `src/`: Core source code
- `config/`: User-editable config and prompt documents (`config_user.yaml`, `persona.md`, `style.md`, `social_tips/`, `self_image/`)
- `assets/`: Static read-only resources (e.g. `voice_example.json`)
- `data/`: Runtime persistent data (SQLite DB, stickers collection)
- `cache/`: Runtime cache — safe to delete (`image/`, `tts/`, `stickers/`)
- `logs/`: Application logs
- `templates/`: Config template (`config.yaml.template`)

## Quick Start (Windows)

Double-click **`start.bat`** to launch the bot.

- **First Run**: It will guide you to set up a Python environment (create venv, use Conda, or system Python) and install dependencies automatically.
- **Subsequent Runs**: It remembers your choice and launches immediately.

To reset the environment configuration:

```cmd
start.bat --reset
```

## Configuration

### 1. API Keys (Security)

Create a `.env` file in the project root (copied from `.env.example`).
**Do not commit this file.**

```bash
cp .env.example .env
# Edit .env and fill in your API keys (e.g. SILICONFLOW_API_KEY)
```

### 2. General Settings

On first launch, `config/config_user.yaml` is automatically generated from `templates/config.yaml.template`. Edit it directly or use the WebUI to configure model providers, bot name, QQ adapter settings, etc.

This file is git-ignored, keeping your personal settings private.

### 3. QQ Adapter

The app only listens on `qq_adapter.host` / `qq_adapter.port`, for example the default `ws://127.0.0.1:8078`.

Configure NapCat or LLoneBot manually so its OneBot v11 reverse WebSocket connects to that URL. Also enable self-message reporting and use the array message format. The app does not read or modify the adapter's local configuration directory.

### 4. Persona

Edit the bot's personality in `config/persona.md`.

### 5. Self Image

Place the bot's avatar image(s) (PNG/JPG/WEBP) in `config/self_image/`. The bot can retrieve these via the `get_self_image` tool when vision is enabled. The folder is git-ignored.

## Development

To run manually without `start.bat`:

```bash
# Activate your environment
# Ensure requirements are installed
python run.py
```

This launcher script adds `src/` to `sys.path` and starts the application.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).
