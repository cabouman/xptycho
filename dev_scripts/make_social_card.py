"""Generate the xptycho social-preview card (Open Graph / GitHub).

The card is 1280x640, the size GitHub and the Open Graph protocol both
accept, so one image serves the GitHub repository preview and the
documentation link preview.  It shows the xPtycho wordmark in the
mbirtorch/mbirjax house style over a field of overlapping glowing probe
spots, the scan pattern ptychography records.  The image renders at
twice the final size and is downsampled once, which anti-aliases the
text and the circles.

The output is written to docs/source/_static/og_card.png, the location
docs/source/conf.py points the og:image meta tag at.
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SCALE = 2                                   # supersample, then downsample
W, H = 1280 * SCALE, 640 * SCALE

TORCH_RED = (238, 42, 74)
LIGHT = (238, 238, 240)
MUTED = (150, 152, 160)
BG_TOP = (17, 18, 22)
BG_BOTTOM = (9, 9, 12)

WORD_FONT = ImageFont.truetype(
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 250 * SCALE)
TAG_FONT = ImageFont.truetype(
    "/System/Library/Fonts/Supplemental/Arial.ttf", 52 * SCALE)
URL_FONT = ImageFont.truetype(
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34 * SCALE)


def background():
    """Returns the dark vertical-gradient background."""
    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        f = y / (H - 1)
        c = tuple(int(BG_TOP[k] + f * (BG_BOTTOM[k] - BG_TOP[k]))
                  for k in range(3))
        for x in range(W):
            px[x, y] = c
    return img


def draw_scan(img):
    """Draws a band of overlapping probe spots along the bottom: soft red
    discs on a jittered grid, with a bright ring on each, blurred for a
    glow."""
    rng = np.random.default_rng(3)
    radius = 46 * SCALE
    step = 58 * SCALE
    rows = [H - 110 * SCALE, H - 168 * SCALE]
    fill = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fill)
    ring = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    for i, y in enumerate(rows):
        x = 90 * SCALE + (i % 2) * step / 2
        while x < W - 60 * SCALE:
            cx = x + rng.normal(0, 3 * SCALE)
            cy = y + rng.normal(0, 3 * SCALE)
            box = [cx - radius, cy - radius, cx + radius, cy + radius]
            fd.ellipse(box, fill=TORCH_RED + (70,))
            rd.ellipse(box, outline=(255, 120, 140, 200), width=3 * SCALE)
            x += step
    fill = fill.filter(ImageFilter.GaussianBlur(6 * SCALE))
    img.paste(fill, (0, 0), fill)
    glow = ring.filter(ImageFilter.GaussianBlur(5 * SCALE))
    img.paste(glow, (0, 0), glow)
    img.paste(ring, (0, 0), ring)


def draw_wordmark(img):
    """Draws 'XPtycho' centered in the upper area: X in torch red, the
    rest in light, with a soft red glow behind the whole word."""
    pieces = [("X", TORCH_RED), ("Ptycho", LIGHT)]
    d = ImageDraw.Draw(img)
    widths = [d.textbbox((0, 0), t, font=WORD_FONT)[2] for t, _ in pieces]
    total = sum(widths)
    x = (W - total) // 2
    y = 70 * SCALE

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.text((x, y), "XPtycho", font=WORD_FONT, fill=TORCH_RED + (110,))
    glow = glow.filter(ImageFilter.GaussianBlur(18 * SCALE))
    img.paste(glow, (0, 0), glow)

    for (t, color), w in zip(pieces, widths):
        d.text((x, y), t, font=WORD_FONT, fill=color)
        x += w


def draw_text(img):
    """Draws the tagline and the documentation URL."""
    d = ImageDraw.Draw(img)
    tag = "Ptychographic reconstruction with PMACE in PyTorch"
    tw = d.textbbox((0, 0), tag, font=TAG_FONT)[2]
    d.text(((W - tw) // 2, 352 * SCALE), tag, font=TAG_FONT, fill=MUTED)

    url = "xptycho.readthedocs.io"
    uw = d.textbbox((0, 0), url, font=URL_FONT)[2]
    d.text((W - 70 * SCALE - uw, 586 * SCALE), url, font=URL_FONT,
           fill=TORCH_RED)


def main():
    img = background().convert("RGBA")
    draw_scan(img)
    draw_wordmark(img)
    draw_text(img)
    img = img.convert("RGB").resize((W // SCALE, H // SCALE),
                                    Image.LANCZOS)
    out = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "..", "docs", "source", "_static", "og_card.png")
    img.save(os.path.normpath(out))
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
