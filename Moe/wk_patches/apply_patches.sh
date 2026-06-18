#!/bin/bash
# Re-apply custom patches to the webknossos package after any upgrade.
# Must be run inside the pixi environment:
#
#   cd /groups/scicompsoft/home/chend/temp/downsample_script/tensorswitch/Moe
#   pixi run bash wk_patches/apply_patches.sh
#
# IMPORTANT: After a major version upgrade (e.g. 3.3→3.5), the patched files
# may be incompatible with the new version's internals.  In that case, manually
# merge the changes from the saved files into the new version before copying.
#
# What these patches fix:
#   cli_convert_zarr.py        — NFS transient read retry (ZSTD_decompressStream
#                                errors, page-cache flush, GC, 5x retry / 10s backoff)
#   client__download_dataset.py — Download retry with exponential backoff (5x,
#                                5s→120s), progress printing, and response validation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WK_DIR=$(python - <<'EOF'
import webknossos, os
print(os.path.dirname(webknossos.__file__))
EOF
)

if [ -z "$WK_DIR" ]; then
    echo "ERROR: could not locate webknossos package"
    exit 1
fi

WK_VER=$(python -c "import pkg_resources; print(pkg_resources.get_distribution('webknossos').version)")
echo "webknossos $WK_VER at: $WK_DIR"

cp "$SCRIPT_DIR/cli_convert_zarr.py"         "$WK_DIR/cli/convert_zarr.py"
echo "  applied: cli/convert_zarr.py"

cp "$SCRIPT_DIR/client__download_dataset.py" "$WK_DIR/client/_download_dataset.py"
echo "  applied: client/_download_dataset.py"

# Clear pycache so Python picks up the new files
find "$WK_DIR/cli" "$WK_DIR/client" -name "*.pyc" -delete 2>/dev/null || true

echo "Done."
