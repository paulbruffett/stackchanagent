#!/usr/bin/env bash
# Convert generated PNG sprites to LVGL v9 RGB565A8 C arrays.
#
# Wraps LVGL's LVGLImage.py. Output symbol == filename, so rocky_busy.png ->
# rocky_busy.c with `rocky_busy`, ready for LV_IMAGE_DECLARE(rocky_busy).
#
# Usage:
#   ./convert.sh                 # convert every PNG in the png/ dir
#   ./convert.sh rocky_busy      # convert one (name without extension)
#   IN=/path/to/pngs OUT=/path/to/c ./convert.sh
#
# Requires: pip3 install pypng  (LVGLImage.py dependency)

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PNGs in, C arrays out (the skin's assets dir, alongside the existing .c files).
IN="${IN:-$here/../../main/stackchan/avatar/skins/rocky/assets/png}"
OUT="${OUT:-$here/../../main/stackchan/avatar/skins/rocky/assets}"

# LVGLImage.py: override with LVGL_IMG_CONV, else use the firmware's lvgl component.
CONV="${LVGL_IMG_CONV:-$here/../../managed_components/lvgl__lvgl/scripts/LVGLImage.py}"

if [[ ! -f "$CONV" ]]; then
  echo "error: LVGLImage.py not found at $CONV" >&2
  echo "       set LVGL_IMG_CONV=/path/to/LVGLImage.py" >&2
  exit 1
fi
if ! python3 -c "import png" 2>/dev/null; then
  echo "error: LVGLImage.py needs pypng -> pip3 install pypng" >&2
  exit 1
fi
if [[ ! -d "$IN" ]]; then
  echo "error: input dir not found: $IN  (run gen_sprites.py first)" >&2
  exit 1
fi

mkdir -p "$OUT"

# Build the file list: explicit names as args, or every PNG in IN.
files=()
if [[ $# -gt 0 ]]; then
  for name in "$@"; do
    f="$IN/${name%.png}.png"
    [[ -f "$f" ]] || { echo "error: no such PNG: $f" >&2; exit 1; }
    files+=("$f")
  done
else
  shopt -s nullglob
  files=("$IN"/*.png)
  shopt -u nullglob
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "no PNGs to convert in $IN" >&2
  exit 1
fi

echo "converting ${#files[@]} sprite(s) -> $OUT"
for f in "${files[@]}"; do
  echo "  $(basename "$f")"
  python3 "$CONV" --ofmt C --cf RGB565A8 -o "$OUT" "$f"
done
echo "done"
