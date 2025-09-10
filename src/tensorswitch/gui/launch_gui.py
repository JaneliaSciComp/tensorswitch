#!/usr/bin/env python3
"""
TensorSwitch GUI Launcher

Simple script to start the TensorSwitch web GUI.
"""

import panel as pn
import os
import sys
from pathlib import Path

# Add project root to path for development
current_dir = Path(__file__).parent  # gui folder
project_root = current_dir.parent.parent.parent  # tensorswitch project root
src_dir = project_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

try:
    from tensorswitch.gui.app import create_simple_app
    print("Successfully imported TensorSwitch GUI (Simple Version)")
except ImportError as e:
    print(f"Failed to import TensorSwitch GUI: {e}")
    print("\nTrying to install dependencies...")
    
    # Try to install Panel if missing
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "panel", "param", "sqlalchemy"])
        print("Installed basic dependencies")
        from tensorswitch.gui.app import create_simple_app
    except Exception as install_error:
        print(f"Could not install dependencies: {install_error}")
        print("\nPlease install manually:")
        print("  pip install panel param sqlalchemy")
        sys.exit(1)


def main():
    """Launch the TensorSwitch GUI."""
    
    print("Starting TensorSwitch GUI...")
    print("Project directory:", project_root)
    
    # Set environment variables
    os.environ.setdefault('PANEL_ALLOW_WEBSOCKET_ORIGIN', '*')
    os.environ.setdefault('PANEL_LOG_LEVEL', 'info')
    
    # Configure Panel
    pn.config.console_output = 'accumulate'
    
    try:
        # Create the simple app
        app = create_simple_app()
        
        # Get port from environment or use default
        port = int(os.environ.get('TENSORSWITCH_GUI_PORT', 5000))
        
        print(f"\nTensorSwitch GUI is starting on port {port}...")
        print(f"For JupyterHub users: http://[your-host]:{port}")
        print(f"For local users: http://localhost:{port}") 
        print("Press Ctrl+C to stop the server")
        print("\n" + "="*60)
        
        # Serve the app
        pn.serve(
            app,
            port=port,
            allow_websocket_origin=["*"],
            show=False,  # Don't auto-open for server environments
            autoreload=False,  # Disable for production
            title="TensorSwitch GUI"
        )
        
    except KeyboardInterrupt:
        print("\nTensorSwitch GUI stopped by user")
    except Exception as e:
        print(f"\nError starting GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()