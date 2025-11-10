#!/usr/bin/env python3
"""
TensorSwitch GUI Launcher

Simple script to start the TensorSwitch web GUI.
Implements Jupyter-style per-user deployment pattern.
"""

import panel as pn
import os
import sys
import socket
import random
import argparse
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


def find_free_port(start=30000, end=40000, max_tries=100):
    """
    Find a free port in the specified range.

    Similar to Jupyter's port selection, but uses range 30000-40000
    as recommended by Goran for multi-user cluster environments.

    Args:
        start: Starting port number (default: 30000)
        end: Ending port number (default: 40000)
        max_tries: Maximum number of attempts (default: 100)

    Returns:
        int: Available port number

    Raises:
        RuntimeError: If no free port found after max_tries attempts
    """
    for _ in range(max_tries):
        port = random.randint(start, end)
        try:
            # Try to bind to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            # Port is already in use, try another
            continue

    raise RuntimeError(f"Could not find a free port after {max_tries} attempts in range {start}-{end}")


def get_hostname():
    """
    Get the full hostname of the current machine.

    Returns:
        str: Hostname (e.g., 'login1.int.janelia.org' or 'localhost')
    """
    try:
        hostname = socket.gethostname()
        # If hostname is just short name, try to get FQDN
        if '.' not in hostname:
            hostname = socket.getfqdn()
        return hostname
    except Exception:
        return 'localhost'


def main():
    """Launch the TensorSwitch GUI."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Launch TensorSwitch GUI - Data format conversion tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-select random port (30000-40000)
  python -m tensorswitch.gui.launch_gui

  # Specify custom port
  python -m tensorswitch.gui.launch_gui --port 35000

  # Specify host and port
  python -m tensorswitch.gui.launch_gui --host 0.0.0.0 --port 35000
        """
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Port to run server on (default: auto-select from 30000-40000)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0 for external access)'
    )

    args = parser.parse_args()

    # Set environment variables
    os.environ.setdefault('PANEL_ALLOW_WEBSOCKET_ORIGIN', '*')
    os.environ.setdefault('PANEL_LOG_LEVEL', 'info')

    # Configure Panel
    pn.config.console_output = 'accumulate'

    try:
        # Create the simple app
        app = create_simple_app()

        # Find a free port or use specified port
        if args.port:
            port = args.port
        else:
            port = find_free_port(start=30000, end=40000)

        # Get hostname
        hostname = get_hostname()

        # Print Jupyter-style startup message
        print("\n" + "="*60)
        print("TensorSwitch GUI Started!")
        print("="*60)
        print("\nTensorSwitch Server is running at:\n")
        print(f"  http://{hostname}:{port}")
        print("\nCopy and paste this URL into your browser")
        print("\nUse Control-C to stop this server")
        print("="*60 + "\n")

        # Serve the app
        pn.serve(
            app,
            port=port,
            address=args.host,
            allow_websocket_origin=["*"],
            show=False,  # Don't auto-open browser (user will copy URL)
            autoreload=False,  # Disable for production
            title="TensorSwitch GUI",
            websocket_origin=f"{hostname}:{port}"
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