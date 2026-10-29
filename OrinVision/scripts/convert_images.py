#!/usr/bin/env python3
"""
convert_images.py - Image helpers for headless (SSH) work.

1) Results -> PNG + one overview image (to view on your PC):
       python3 scripts/convert_images.py results
   Converts every .pgm/.ppm in the folder to .png and creates
   overview.png showing all pipeline stages side by side.

2) Your photo -> PPM (input for orin_vision):
       python3 scripts/convert_images.py --to-ppm photo.jpg
   Creates photo.ppm next to it.

Requires Pillow:  sudo apt install python3-pil
"""
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Pillow not installed. Run: sudo apt install python3-pil")

STAGE_FILES = [
    ("0_input.ppm", "input"),
    ("1_gray.pgm", "1. grayscale"),
    ("2_equalized.pgm", "2. equalized"),
    ("3_blurred.pgm", "3. blurred"),
    ("4_edges.pgm", "4. edges"),
    ("5_canny.pgm", "5. canny edges"),
    ("6_harris.ppm", "6. harris corners"),
    ("7_fast.ppm", "7. FAST corners"),
    ("8_orb.ppm", "8. ORB features"),
]


def to_png(folder: Path):
    files = sorted(list(folder.glob("*.pgm")) + list(folder.glob("*.ppm")))
    if not files:
        sys.exit(f"No .pgm/.ppm files in {folder}")
    for f in files:
        out = f.with_suffix(".png")
        Image.open(f).save(out)
        print(f"  {out}")
    make_overview(folder)


def make_overview(folder: Path, thumb_width: int = 640):
    tiles = []
    for name, label in STAGE_FILES:
        path = folder / name
        if path.exists():
            im = Image.open(path).convert("RGB")
            h = round(im.height * thumb_width / im.width)
            tiles.append((im.resize((thumb_width, h)), label))
    if not tiles:
        return
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    th = tiles[0][0].height
    pad = 28
    sheet = Image.new("RGB", (cols * thumb_width, rows * (th + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(tiles):
        x = (i % cols) * thumb_width
        y = (i // cols) * (th + pad)
        draw.text((x + 8, y + 7), label, fill="black")
        sheet.paste(im, (x, y + pad))
    out = folder / "overview.png"
    sheet.save(out)
    print(f"  {out}   <- all stages in one picture")


def to_ppm(files):
    for name in files:
        src = Path(name)
        out = src.with_suffix(".ppm")
        Image.open(src).convert("RGB").save(out)
        print(f"  {out}")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    if args[0] == "--to-ppm":
        to_ppm(args[1:])
    else:
        to_png(Path(args[0]))


if __name__ == "__main__":
    main()
