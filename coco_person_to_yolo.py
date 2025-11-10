
#!/usr/bin/env python3
"""
coco_person_to_yolo.py

Usage (from repo root, inside your activated venv):
    python scripts/coco_person_to_yolo.py \
      --coco-dir "data/coco2017" \
      --out-dir "data/person_val" \
      --limit 500

This script:
 - reads instances_val2017.json
 - extracts annotations with category_id == 1 (person)
 - copies images that contain persons to out_dir/images/
 - writes YOLO .txt files in out_dir/labels/ with lines: "0 x_center y_center width height"
   (normalized to [0,1] relative to image width/height). Class 0 == person.
"""
import json
import os
import shutil
from pathlib import Path
from PIL import Image
import argparse
from collections import defaultdict

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def convert_bbox_coco_to_yolo(box, img_w, img_h):
    # COCO bbox: [x_min, y_min, width, height] (absolute pixels)
    x_min, y_min, w, h = box
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0
    # normalize
    return x_center / img_w, y_center / img_h, w / img_w, h / img_h

def main(args):
    coco_dir = Path(args.coco_dir)
    ann_file = coco_dir / "annotations" / "instances_val2017.json"
    imgs_dir = coco_dir / "val2017"
    out_dir = Path(args.out_dir)

    if not ann_file.exists():
        raise FileNotFoundError(f"Cannot find annotations file: {ann_file}")

    ensure_dir(out_dir)
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    ensure_dir(images_out)
    ensure_dir(labels_out)

    data = json.load(open(ann_file, "r", encoding="utf8"))
    # map image_id -> filename & size
    images = {img["id"]: img for img in data["images"]}

    # collect only person annotations (COCO category_id == 1)
    person_anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("category_id") == 1 and ann.get("iscrowd", 0) == 0:
            img_id = ann["image_id"]
            person_anns_by_image[img_id].append(ann)

    # Optionally limit number of images (for fast experiments)
    image_ids = list(person_anns_by_image.keys())
    if args.limit and args.limit > 0:
        image_ids = image_ids[: args.limit]

    copied = 0
    labels_written = 0
    for img_id in image_ids:
        img_meta = images.get(img_id)
        if img_meta is None:
            continue
        file_name = img_meta["file_name"]
        src_path = imgs_dir / file_name
        if not src_path.exists():
            # Some users extract to a different folder; skip missing files
            print(f"Warning: image file not found: {src_path}")
            continue

        # copy image
        dst_img_path = images_out / file_name
        if not dst_img_path.exists():
            shutil.copy2(src_path, dst_img_path)
        copied += 1

        # open to get size (fallback if width/height not present)
        img_w = img_meta.get("width")
        img_h = img_meta.get("height")
        if (img_w is None) or (img_h is None):
            try:
                with Image.open(src_path) as im:
                    img_w, img_h = im.size
            except Exception as e:
                print(f"Could not open image to read size {src_path}: {e}")
                continue

        # build YOLO label file for this image
        label_lines = []
        anns = person_anns_by_image[img_id]
        for ann in anns:
            # ann['bbox'] is [x_min, y_min, w, h]
            x_c, y_c, w_n, h_n = convert_bbox_coco_to_yolo(ann["bbox"], img_w, img_h)
            # clamp to [0,1]
            x_c = min(max(x_c, 0.0), 1.0)
            y_c = min(max(y_c, 0.0), 1.0)
            w_n = min(max(w_n, 0.0), 1.0)
            h_n = min(max(h_n, 0.0), 1.0)
            # class 0 = person
            label_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        label_path = labels_out / (Path(file_name).stem + ".txt")
        with open(label_path, "w", encoding="utf8") as f:
            f.write("\n".join(label_lines))
        labels_written += len(label_lines)

    print("Done.")
    print(f"Images copied: {copied}")
    print(f"Person annotations written (total boxes): {labels_written}")
    print(f"YOLO dataset ready in: {out_dir.resolve()}")
    print(f" - images: {images_out.resolve()}")
    print(f" - labels: {labels_out.resolve()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", type=str, default="data/coco2017", help="Path to coco2017 folder (containing 'val2017' and 'annotations').")
    parser.add_argument("--out-dir", type=str, default="data/person_val", help="Output folder for YOLO-format person dataset.")
    parser.add_argument("--limit", type=int, default=0, help="Optional: limit to N images (0 = no limit).")
    args = parser.parse_args()
    main(args)
