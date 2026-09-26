"""Generate the xptycho logos in the mbirtorch/mbirjax house style:
bold sans wordmark, black base, colored accent, glossy reflection.

The logos are written into docs/source/_static/, where the docs use
them, so running this script updates the real assets in place.
"""

import os

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# The docs static directory, resolved from this script's own location so
# the output goes to the same place no matter where the script is run.
STATIC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..", "docs", "source", "_static"))

FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 260)
CANVAS = (1900, 560)
BASELINE_Y = 60


def text_layer(pieces):
    """Render the wordmark pieces, each with its own solid color."""
    img = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    x = 60
    for piece, color in pieces:
        mask = Image.new("L", CANVAS, 0)
        d = ImageDraw.Draw(mask)
        d.text((x, BASELINE_Y), piece, font=FONT, fill=255)
        bbox = d.textbbox((x, BASELINE_Y), piece, font=FONT)
        img.paste(Image.new("RGBA", CANVAS, color + (255,)), (0, 0), mask)
        x = bbox[2] + 6
    return img


def add_reflection(img):
    """Mirror the wordmark below with a fading, slightly blurred copy."""
    bbox = img.getbbox()
    word = img.crop(bbox)
    refl = word.transpose(Image.FLIP_TOP_BOTTOM)
    refl = refl.resize((refl.width, int(refl.height * 0.85)))
    refl = refl.filter(ImageFilter.GaussianBlur(1.5))
    fade = Image.new("L", refl.size, 0)
    fp = fade.load()
    for y in range(refl.height):
        a = max(0, int(150 * (1 - y / (refl.height * 0.85))))
        for x in range(refl.width):
            fp[x, y] = a
    a = refl.split()[3]
    refl.putalpha(ImageChops.multiply(a, fade))
    margin = 40
    out = Image.new("RGBA",
                    (word.width + 2 * margin,
                     word.height + refl.height + margin + 20),
                    (0, 0, 0, 0))
    out.paste(word, (margin, 10), word)
    out.paste(refl, (margin, 10 + word.height + 8), refl)
    return out


BLACK = (20, 20, 20)
LIGHT = (235, 235, 235)
TORCH_RED = (238, 42, 74)

# X in torch red ties xptycho to the mbirtorch family.  "Ptycho" is dark
# for the light-background logo and light for the dark-background
# logo, so it stays visible in both documentation themes.
variants = {
    "logo": [("X", TORCH_RED), ("Ptycho", BLACK)],
    "logo_dark": [("X", TORCH_RED), ("Ptycho", LIGHT)],
}

for name, spec in variants.items():
    img = add_reflection(text_layer(spec))
    out = os.path.join(STATIC, f"{name}.png")
    img.save(out)
    print("wrote", out)
