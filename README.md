# AIcarusForQQ

QQ AI Bot with LLM integration (Restructured).

## Structure

- `src/`: Core source code (`main.py`, adapters, handlers)
- `config/`: Configuration files (`config.yaml`, schemas)
- `data/`: Runtime data (Persona, SQLite DB, Self Image)
- `logs/`: Application logs
- `archive/`: Legacy code (e.g. `provider_old.py`)

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

- **Default Config**: The project comes with a safe default at `config/config.yaml`.
- **User Overrides**: To customize settings (e.g., model name, temperature, bot name), create a file named **`config_user.yaml`** in the project root.

  Model backends are now configured through named OpenAI-compatible profiles. You can select a `profile` and extend `openai_profiles` with your own compatible endpoint, API key env var, and default model.

  Any settings defined in `config_user.yaml` will override `config/config.yaml`. This file is git-ignored, keeping your personal tweaks private.

  **Example `config_user.yaml`**:

  ```yaml
  bot_name: "Shadow"
  model: "Pro/zai-org/GLM-5"
  timezone: "Asia/Tokyo"
  ```

### 3. Persona

Edit the bot's personality in `data/persona.md`.

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
