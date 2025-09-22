#!/usr/bin/env python3
import argparse, json, os, glob, uuid, re, sys
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util

from label_studio_converter import brush
from lxml import etree

from label_studio_sdk.client import LabelStudio as LSClient

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def log(x): print(x, flush=True)

# ---------- helpers ----------
def coco_rle_to_ls_rle(coco_rle):
    H, W = coco_rle["size"]
    counts = coco_rle.get("counts")
    if isinstance(counts, str):
        rle = {"size": [H, W], "counts": counts.encode("ascii")}
    elif isinstance(counts, (bytes, bytearray)) or isinstance(counts, list):
        rle = {"size": [H, W], "counts": counts}
    else:
        raise ValueError("bad RLE")
    m = mask_util.decode(rle)
    if m.ndim == 3: m = m[:, :, 0]
    m = (m.astype(np.uint8) * 255)
    return brush.mask2rle(m)

def bbox_xyxy_to_ls_pct(x1,y1,x2,y2,W,H):
    return {"x":100.0*x1/W,"y":100.0*y1/H,"width":100.0*(x2-x1)/W,"height":100.0*(y2-y1)/H,"rotation":0.0}

def find_image_for_json(json_path: str) -> Optional[str]:
    base = os.path.basename(json_path); dirn = os.path.dirname(json_path)
    m = re.match(r"^(?P<stem>.+)(?P<ext>\.[A-Za-z0-9]+)_.+\.json$", base)
    if m:
        c = os.path.join(dirn, m.group("stem") + m.group("ext"))
        if os.path.exists(c): return c
    stem, _ = os.path.splitext(base)
    for ext in IMG_EXTS:
        c = os.path.join(dirn, stem + ext)
        if os.path.exists(c): return c
    parts = base.split(".")
    if len(parts) >= 3:
        c = os.path.join(dirn, ".".join(parts[:2]))
        if os.path.exists(c): return c
    return None

def norm_filename(p: str) -> str:
    b = os.path.basename(p)
    if "-" in b:
        first, rest = b.split("-", 1)
        if re.fullmatch(r"[0-9a-fA-F]{6,64}", first):
            return rest.lower()
    return b.lower()

# ---------- parse project labeling config ----------
def parse_label_config(xml_str: str):
    """
    Returns:
      data_key (str), to_name (str),
      rect_from (Optional[str]), brush_from (Optional[str]),
      allowed_labels (set[str])
    """
    root = etree.fromstring(xml_str.encode("utf-8"))
    img_tag = root.xpath("//Image")[0]
    to_name = img_tag.get("name")
    val = img_tag.get("value")  # like "$image"
    data_key = val[1:] if val and val.startswith("$") else val

    rect = root.xpath("//RectangleLabels")
    brushl = root.xpath("//BrushLabels")
    rect_from = rect[0].get("name") if rect else None
    brush_from = brushl[0].get("name") if brushl else None

    allowed = set()
    for ctrl in rect + brushl:
        for c in ctrl.xpath(".//Label"):
            v = c.get("value")
            if v: allowed.add(v)
    return data_key, to_name, rect_from, brush_from, allowed

# ---------- build results ----------
def build_results(dinox: dict, W:int, H:int,
                  rect_from: Optional[str], brush_from: Optional[str], to_name: str,
                  label_map) -> Tuple[List[dict], List[float]]:
    out, scores = [], []
    for o in dinox.get("objects", []):
        raw_label = str(o.get("category","object"))
        label = label_map(raw_label)
        if label is None:
            continue  # skip unknown labels
        sc = float(o.get("score", 0.0)); scores.append(sc)

        if rect_from and "bbox" in o and isinstance(o["bbox"], (list,tuple)) and len(o["bbox"])==4:
            x1,y1,x2,y2 = o["bbox"]
            val = bbox_xyxy_to_ls_pct(x1,y1,x2,y2,W,H)
            val["rectanglelabels"] = [label]
            out.append({
                "id": str(uuid.uuid4()), "type":"rectanglelabels",
                "from_name": rect_from, "to_name": to_name,
                "original_width": W, "original_height": H,
                "image_rotation": 0, "value": val
            })

        m = o.get("mask")
        if brush_from and isinstance(m, dict) and m.get("format","").lower().startswith("coco"):
            out.append({
                "id": str(uuid.uuid4()), "type":"brushlabels",
                "from_name": brush_from, "to_name": to_name,
                "original_width": W, "original_height": H,
                "image_rotation": 0,
                "value": {"format":"rle","rle": coco_rle_to_ls_rle(m),"brushlabels":[label]}
            })
    return out, scores

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=False, help="Directory to save one JSON per image (optional)")
    ap.add_argument("--model_version", default="DINO-X-1.0")
    ap.add_argument("--ls_url", default="http://10.100.52.107:8080")
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--project_id", type=int, required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not files:
        log("No JSON files."); sys.exit(1)

    # Ensure output directory exists if provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # SDK client & project
    ls = LSClient(base_url=args.ls_url, api_key=args.api_key)
    project = ls.projects.get(id=args.project_id)

    # Discover config (to_name/from_name/labels) from project
    data_key, to_name, rect_from, brush_from, allowed = parse_label_config(project.label_config)

    # Build task map: normalized filename -> (task_id, task_image_url)
    task_map: Dict[str, Tuple[int,str]] = {}
    for t in ls.tasks.list(project=project.id, include=["id","data"]):
        img_url = t.data.get(data_key)
        if not img_url: continue
        task_map[norm_filename(img_url)] = (t.id, img_url)
    log(f"Loaded {len(task_map)} tasks from project {project.id}")

    # label normalizer/mapping
    low_allowed = {a.lower(): a for a in allowed}
    unmatched_labels = {}
    total_unmatched_labels = 0
    def label_map(raw: str) -> Optional[str]:
        nonlocal total_unmatched_labels
        s = raw.strip()
        if s in allowed: return s
        key = s.lower().replace("_"," ").strip()
        if key in low_allowed: return low_allowed[key]
        # simple plural/singular tweaks
        if key.endswith("s") and key[:-1] in low_allowed: return low_allowed[key[:-1]]
        if key + "s" in low_allowed: return low_allowed[key+"s"]
        # Track unmatched label
        unmatched_labels[s] = unmatched_labels.get(s, 0) + 1
        total_unmatched_labels += 1
        return None  # skip if not in project

    wrote = posted = unmatched = 0

    for jf in files:
        img_local = find_image_for_json(jf)
        if not img_local:
            log(f"Skip: no image for {jf}"); unmatched += 1; continue

        try:
            with open(jf, "r", encoding="utf-8") as f:
                dinox = json.load(f)
        except Exception as e:
            log(f"Skip bad JSON {jf}: {e}"); continue

        # image size
        try:
            with Image.open(img_local) as im: W,H = im.size
        except Exception:
            # last resort: read from first mask
            any_obj = next((o for o in dinox.get("objects", []) if "mask" in o), None)
            if any_obj and "size" in any_obj["mask"]: H,W = any_obj["mask"]["size"]
            else:
                log(f"Skip: cannot determine size for {jf}"); continue

        results, scores = build_results(dinox, W, H, rect_from, brush_from, to_name, label_map)
        if not results: continue
        avg_score = float(np.mean(scores)) if scores else 0.0

        # match to existing task by filename (no Windows path!)
        key = norm_filename(img_local)
        match = task_map.get(key)
        if not match:
            log(f"No task match for {img_local} (normalized '{key}')")
            unmatched += 1
            continue
        task_id, task_img_url = match

        # POST prediction to the existing task (no `data` in payload)
        ls.predictions.create(
            task=task_id,
            model_version=args.model_version,
            score=avg_score,
            result=results
        )
        posted += 1

        # Write one JSON per image to output_dir if requested
        if args.output_dir:
            output_filename = os.path.splitext(os.path.basename(jf))[0] + ".json"
            output_path = os.path.join(args.output_dir, output_filename)
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump({
                    "data": {data_key: task_img_url},
                    "predictions": [{
                        "model_version": args.model_version,
                        "score": avg_score,
                        "result": results
                    }]
                }, out_f, ensure_ascii=False, indent=2)
            wrote += 1

    if args.output_dir:
        log(f"Wrote {wrote} JSON files to {args.output_dir}")
    log(f"Posted {posted} predictions. Unmatched: {unmatched}")

    # Print unmatched label stats
    if unmatched_labels:
        log("Unmatched label counts:")
        for label, count in unmatched_labels.items():
            log(f"{label} - {count}")
        log(f"All unmatched - {total_unmatched_labels}")

if __name__ == "__main__":
    main()
