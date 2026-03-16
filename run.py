#!/usr/bin/env python3
"""
AIcarusForQQ Launcher
This script sets up the environment and launches the main application.
"""
import os
import sys

def main():
    # Set the base directory to the location of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the src directory to sys.path so modules can be imported
    src_dir = os.path.join(base_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Change working directory to ensure relative paths work as expected
    # (Although we are patching config_loader to be smarter)
    os.chdir(base_dir)

    print(f"🚀 Launching AIcarusForQQ from {base_dir}...")
    
    try:
        from src.main import app

        # On import, src/main.py is executed and loads the configuration.
        # We can access the loaded config directly from the main module.
        from src.main import app, config
        
        try:
            server_config = config.get("server", {})
            # Use 5000 as default to be consistent with main.py
            port = server_config.get("port", 5000)
            host = server_config.get("host", "127.0.0.1")
        except Exception as e:
            print(f"⚠️  Could not load config for port/host: {e}")
            port = 5000
            host = "127.0.0.1"
            
        print(f"🌍 Server starting at http://{host}:{port}")
        app.run(host=host, port=port)
        
    except ImportError as e:
        print(f"❌ Error: Could not import application modules. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
