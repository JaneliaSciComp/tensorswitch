"""_quiet_capture reports real warnings to the MCP caller and drops harmless library noise."""
import sys
import warnings

from tensorswitch_v2 import mcp_server as m


def test_real_warnings_are_kept_and_benign_ones_dropped():
    with m._quiet_capture() as notes:
        warnings.warn("voxel size looks wrong", UserWarning)
        warnings.warn("Casting invalid PixelsID '0' to 'Pixels:0'", UserWarning)
        print("WARNING Casting invalid PixelsID '1' to 'Pixels:1'", file=sys.stderr)
        print("WARNING something real", file=sys.stderr)
    assert notes == ["voxel size looks wrong", "WARNING something real"]
