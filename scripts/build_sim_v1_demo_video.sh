#!/usr/bin/env bash
# Assemble already-saved local simulation videos; never reruns policies or accesses hardware.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${1:-$ROOT/artifacts/sim-v1-demo-20260925T151219Z-Rdivk1}"
OUT="${2:-$ROOT/artifacts/sim-v1-release-current-20260925/demo-65s.mp4}"
for f in "$SOURCE/run/dual-camera.mp4" "$SOURCE/act/dual-camera.mp4" "$SOURCE/smolvla/dual-camera.mp4"; do
  test -s "$f" || { echo "Missing saved video: $f" >&2; exit 2; }
done
mkdir -p "$(dirname -- "$OUT")"
ffmpeg -y -loglevel error \
  -f lavfi -i 'color=c=0x101821:s=1280x480:r=10:d=5' \
  -i "$SOURCE/run/dual-camera.mp4" \
  -i "$SOURCE/act/dual-camera.mp4" \
  -i "$SOURCE/smolvla/dual-camera.mp4" \
  -f lavfi -i 'color=c=0x101821:s=1280x480:r=10:d=6' \
  -filter_complex "[0:v]format=yuv420p,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='SO101  |  Pure simulation':fontcolor=white:fontsize=52:x=80:y=150,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Tape roll to green mat':fontcolor=0x87d6ff:fontsize=32:x=80:y=235[intro];[1:v]fps=10,format=yuv420p,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='FIXED SCRIPT  |  success':fontcolor=white:fontsize=31:box=1:boxcolor=black@0.65:x=24:y=24[fixed];[2:v]fps=10,format=yuv420p,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='ACT  |  timeout, no tape contact':fontcolor=white:fontsize=31:box=1:boxcolor=black@0.65:x=24:y=24[act];[3:v]fps=10,format=yuv420p,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='SmolVLA  |  timeout, no tape contact':fontcolor=white:fontsize=31:box=1:boxcolor=black@0.65:x=24:y=24[vla];[4:v]format=yuv420p,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Development scene only':fontcolor=white:fontsize=45:x=80:y=150,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Physical alignment and formal comparison remain open':fontcolor=0x87d6ff:fontsize=26:x=80:y=235[outro];[intro][fixed][act][vla][outro]concat=n=5:v=1:a=0[v]" \
  -map '[v]' -an -c:v libx264 -crf 23 -pix_fmt yuv420p -movflags +faststart "$OUT"
printf '%s\n' "$OUT"
