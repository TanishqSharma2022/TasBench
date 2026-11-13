#!/usr/bin/env python3
"""
Helper to draw bounding boxes / polygons onto table images using PIL.
Handles:
- pixel bboxes: [x1, y1, x2, y2]
- normalized bboxes: [x1, y1, x2, y2] where values are in [0, 1]
- 8-point polygons: [x1, y1, x2, y2, x3, y3, x4, y4]
- 4-point list of points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
"""
import os
from PIL import Image, ImageDraw, ImageFont

def draw_bboxes_on_image(image_path, bboxes, output_path, outline="#22c55e", width=2):
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for idx, box in enumerate(bboxes):
        if box is None or len(box) == 0:
            continue
        
        # Check if 4 numbers (x1, y1, x2, y2)
        if len(box) == 4 and not isinstance(box[0], (list, tuple)):
            x1, y1, x2, y2 = box
            # If normalized in [0, 1]
            if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0:
                x1, x2 = x1 * w, x2 * w
                y1, y2 = y1 * h, y2 * h
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        elif len(box) == 8:
            pts_raw = [(box[i], box[i+1]) for i in range(0, 8, 2)]
            max_v = max(max(p) for p in pts_raw)
            if max_v <= 1.0:
                pts = [(px * w, py * h) for px, py in pts_raw]
            else:
                pts = pts_raw
        elif len(box) == 4 and isinstance(box[0], (list, tuple)):
            pts_raw = [(p[0], p[1]) for p in box]
            max_v = max(max(p) for p in pts_raw)
            if max_v <= 1.0:
                pts = [(px * w, py * h) for px, py in pts_raw]
            else:
                pts = pts_raw
        else:
            continue

        # Draw box polygon
        draw.polygon(pts, outline=outline, width=width)
        
        # Draw cell index
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        draw.text((cx - 4, cy - 4), str(idx + 1), fill="#ef4444", font=font)

    # Composite overlay
    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(output_path)
    print(f"Saved visualization with {len(bboxes)} bboxes to {output_path}")

if __name__ == "__main__":
    pass
