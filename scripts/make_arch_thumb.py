#!/usr/bin/env python3
"""
Генератор мініатюри кадру для figures/architecture.html (блок Detections).

Малює рамку(и) детекції на реальному кадрі LaSOT, обрізає навколо цілі,
масштабує до 300px і друкує base64 JPEG для вставки як data-URI у SVG.

Поточний кадр у схемі: elephant-18, кадр 498 (велика чітка ціль).
Щоб замінити — зміни VIDEO / FRAME / BOXES нижче й онови href у architecture.html.

    python scripts/make_arch_thumb.py            # друкує b64 + зберігає preview
"""
from PIL import Image, ImageDraw
import base64
import io

VIDEO = 'elephant-18'
FRAME = 498                       # індекс у groundtruth (файл = FRAME+1)
# рамки для малювання: (x1, y1, x2, y2, колір). None → взяти GT цього кадру синім.
BOXES = None
BLUE = (42, 111, 208)
RED = (210, 59, 65)
THUMB_W = 300
CROP_SCALE = 2.6                  # у скільки разів вікно кадрування ширше за ціль

ROOT = '/home/peoly/datasets/lasot/test'
OUT_PREVIEW = 'figures/architecture_thumb_%s.jpg'


def main():
    cat = VIDEO.rsplit('-', 1)[0]
    d = f'{ROOT}/lasot_test_{cat}/{VIDEO}'
    gt = [[float(v) for v in l.replace('\t', ',').split(',')]
          for l in open(f'{d}/groundtruth.txt')]
    x, y, w, h = gt[FRAME]
    im = Image.open(f'{d}/img/{FRAME + 1:08d}.jpg').convert('RGB')
    W, H = im.size

    cx, cy = x + w / 2, y + h / 2
    cw = w * CROP_SCALE
    ch = cw * 2 / 3
    if ch < h * 1.5:
        ch = h * 1.5
        cw = ch * 3 / 2
    x0, y0 = max(0, int(cx - cw / 2)), max(0, int(cy - ch / 2))
    x1, y1 = min(W, int(cx + cw / 2)), min(H, int(cy + ch / 2))
    crop = im.crop((x0, y0, x1, y1))

    boxes = BOXES if BOXES is not None else [(x, y, x + w, y + h, BLUE)]
    dr = ImageDraw.Draw(crop)
    for bx0, by0, bx1, by1, color in boxes:
        b = (bx0 - x0, by0 - y0, bx1 - x0, by1 - y0)
        for wi in range(3):
            dr.rectangle([b[0] - wi, b[1] - wi, b[2] + wi, b[3] + wi], outline=color)

    tw = THUMB_W
    th = int(crop.height * tw / crop.width)
    crop = crop.resize((tw, th), Image.LANCZOS)

    buf = io.BytesIO()
    crop.save(buf, 'JPEG', quality=84, optimize=True)
    raw = buf.getvalue()
    crop.save(OUT_PREVIEW % VIDEO, 'JPEG', quality=84)

    b64 = base64.b64encode(raw).decode()
    print(f'# {VIDEO} кадр {FRAME}: {tw}x{th}, aspect {th / tw:.3f}, {len(b64) / 1024:.1f} KB b64')
    print(f'# у architecture.html: href="data:image/jpeg;base64,{{...}}"; clipPath/rect під aspect')
    print(b64)


if __name__ == '__main__':
    main()
