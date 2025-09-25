
import argparse, os, sys, json, glob, uuid, re
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import PIL.Image as PILImage
except Exception:
    from PIL import Image as PILImage

import pycocotools.mask as mask_util

# DINO-X SDK

from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.image_resizer import image_to_base64
from dds_cloudapi_sdk.tasks.v2_task import V2Task

# Label Studio SDK + helpers
from label_studio_sdk.client import LabelStudio as LSClient
from label_studio_converter import brush
from lxml import etree

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

# ---------- mapping (lowercase keys and values) ----------
MAPPING = {
    "car": "civilian vehicle","van": "civilian vehicle","bus": "civilian vehicle",
    "truck": "civilian vehicle","motorcycle": "civilian vehicle","boat": "civilian vehicle",
    "bicycle": "civilian vehicle","minivan": "civilian vehicle","nissan": "civilian vehicle",
    "vehicle": "civilian vehicle","ship": "civilian vehicle","bike": "civilian vehicle",
    "jeep": "civilian vehicle","scooter": "civilian vehicle","taxi": "civilian vehicle","suv": "civilian vehicle",
    "bird": "animal","dog": "animal","mammal": "animal","chicken": "animal","elephant": "animal",
    "giraffe": "animal","goose": "animal","horse": "animal","sheep": "animal","animal": "animal",
    "fish": "animal","duck": "animal","pig": "animal","bear": "animal",
    "person": "citizen","pedestrian": "citizen","woman": "citizen","people": "citizen",
    "mound": "water","pool": "water","lake": "water","river": "water","puddle": "water",
    "flower": "plant",
    "house": "building","skyscraper": "building","bridge": "building",
    "hurdle": "barrier",
    "garbage": "debris","rubble": "debris",
    "bulldozer": "excavator",
    "glasses": "protective glasses",
    "sneaker": "boot","footwear": "boot",
    "handbag": "bag","backpack": "bag",
    "hose": "fire hose",
    "fire": "flame",
    "sinkhole": "hole in the ground","manhole": "hole in the ground",
}

def log(x): print(x, flush=True)

# ---------- DINO-X API ----------
def call_dinox_api(token: str, image_path: str, mask_format: str = "coco_rle",
                   bbox_threshold: float = 0.25, iou_threshold: float = 0.8) -> dict:
    client = Client(Config(token))
    image_b64 = image_to_base64(image_path)
    api_path = "/v2/task/dinox/detection"
    api_body = {
        "model": "DINO-X-1.0",
        "image": image_b64,
        "prompt": {"type": "universal"},
        "targets": ["bbox", "mask"],
        "mask_format": mask_format,
        "bbox_threshold": bbox_threshold,
        "iou_threshold": iou_threshold,
    }
    task = V2Task(api_path=api_path, api_body=api_body)
    client.run_task(task)
    if not task.result or "objects" not in task.result:
        return {"objects": []}
    return task.result

def apply_label_mapping(seg_result: dict, mapping: dict = MAPPING) -> dict:
    for obj in seg_result.get("objects", []):
        raw = str(obj.get("category", "")).lower()
        obj["category"] = mapping.get(raw, raw)
    return seg_result

# ---------- Label Studio helpers ----------
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

def norm_filename(p: str) -> str:
    b = os.path.basename(p)
    if "-" in b:
        first, rest = b.split("-", 1)
        if re.fullmatch(r"[0-9a-fA-F]{6,64}", first):
            return rest.lower()
    return b.lower()

def parse_label_config(xml_str: str):
    root = etree.fromstring(xml_str.encode("utf-8"))
    img_tag = root.xpath("//Image")[0]
    to_name = img_tag.get("name")
    val = img_tag.get("value")
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

def build_results(dinox: dict, W:int, H:int,
                  rect_from: Optional[str], brush_from: Optional[str], to_name: str,
                  label_map) -> Tuple[List[dict], List[float]]:
    out, scores = [], []
    for o in dinox.get("objects", []):
        raw_label = str(o.get("category","object"))
        label = label_map(raw_label)
        if label is None:
            continue
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
        if brush_from and isinstance(m, dict) and str(m.get("format","")).lower().startswith("coco"):
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
    ap = argparse.ArgumentParser(description="Run DINO-X on images and push predictions to Label Studio tasks.")
    ap.add_argument("--token", required=True, help="DINO-X API token")
    ap.add_argument("--input_dir", required=True, help="Directory with images already imported as LS tasks")
    ap.add_argument("--glob", default="*.jpg,*.jpeg,*.png,*.bmp,*.tif,*.tiff,*.webp", help="Comma-separated patterns")
    ap.add_argument("--bbox_threshold", type=float, default=0.25)
    ap.add_argument("--iou_threshold", type=float, default=0.8)
    ap.add_argument("--mask_format", default="coco_rle", choices=["coco_rle"])
    ap.add_argument("--ls_url", default="http://10.100.52.107:8080")
    ap.add_argument("--api_key", required=True, help="Label Studio API key")
    ap.add_argument("--project_id", type=int, required=True)
    ap.add_argument("--model_version", default="DINO-X-1.0")
    ap.add_argument("--output_dir", required=False, help="Optional dir to save one JSON per image")
    ap.add_argument("--unmatched_csv", required=False, default="unmatched_labels.csv", help="CSV file to append unmatched label counts")
    ap.add_argument("--allow_upload", action="store_true", help="Allow uploading images to Label Studio if not found")
    ap.add_argument("--force-upload", action="store_true", help="Force upload predictions even if JSON exists for this image")
    args = ap.parse_args()

    # enumerate images
    patterns = [p.strip() for p in args.glob.split(",") if p.strip()]
    img_files = []
    for pat in patterns:
        img_files.extend(glob.glob(os.path.join(args.input_dir, pat)))
    img_files = sorted({f for f in img_files if os.path.isfile(f)})
    if not img_files:
        log("No images found."); sys.exit(1)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # LS client and project
    ls = LSClient(base_url=args.ls_url, api_key=args.api_key)
    project = ls.projects.get(id=args.project_id)
    data_key, to_name, rect_from, brush_from, allowed = parse_label_config(project.label_config)

    # build task map: normalized filename -> (task_id, task_image_url)
    task_map: Dict[str, Tuple[int,str]] = {}
    for t in ls.tasks.list(project=project.id, include=["id","data"]):
        img_url = t.data.get(data_key)
        if not img_url: continue
        task_map[norm_filename(img_url)] = (t.id, img_url)
    log(f"Loaded {len(task_map)} tasks from project {project.id}")

    # label mapper respecting the project's allowed labels
    low_allowed = {a.lower(): a for a in allowed}
    unmatched_labels = {}
    total_unmatched = 0
    def label_map(raw: str) -> Optional[str]:
        nonlocal total_unmatched
        s = raw.strip()
        if s in allowed: return s
        key = s.lower().replace("_"," ").strip()
        if key in low_allowed: return low_allowed[key]
        if key.endswith("s") and key[:-1] in low_allowed: return low_allowed[key[:-1]]
        if key + "s" in low_allowed: return low_allowed[key+"s"]
        unmatched_labels[s] = unmatched_labels.get(s, 0) + 1
        total_unmatched += 1
        return None

    posted = wrote = skipped = 0

    for img_path in img_files:
        norm_base = norm_filename(img_path)
        json_exists = False
        json_path = None
        if args.output_dir:
            json_path = os.path.join(args.output_dir, os.path.splitext(norm_base)[0] + ".json")
            if os.path.exists(json_path):
                json_exists = True

        if json_exists and not args.force_upload:
            log(f"Skip upload for {img_path} (normalized '{norm_base}') - JSON already exists. Use --force-upload to override.")
            skipped += 1
            continue

        if json_exists and args.force_upload:
            # Load results from JSON, skip DINO-X call
            print(f"Load results from {json_path} for {img_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                pred_json = json.load(f)
            # Use the first prediction in the file
            predictions = pred_json.get("predictions", [])
            if not predictions:
                log(f"No predictions in JSON for {img_path}")
                skipped += 1
                continue
            pred = predictions[0]
            results = pred.get("result", [])
            avg_score = pred.get("score", 0.0)
            # Try to get task_img_url from JSON data
            task_img_url = pred_json.get("data", {}).get(data_key, None)
            if not task_img_url:
                # fallback to task map
                key = norm_filename(img_path)
                match = task_map.get(key)
                if not match:
                    candidates = [(k, v) for k, v in task_map.items() if key in k]
                    if len(candidates) == 1:
                        match = candidates[0][1]
                        log(f"Fuzzy match: '{key}' found in '{candidates[0][0]}'")
                    elif len(candidates) > 1:
                        match = candidates[0][1]
                        log(f"Multiple fuzzy matches for '{key}': {[c[0] for c in candidates]}, using first.")
                    else:
                        log(f"No task match for {img_path} (normalized '{key}')")
                        skipped += 1
                        continue
                task_id, task_img_url = match
            else:
                # find task_id by matching task_img_url
                match = None
                for k, v in task_map.items():
                    if v[1] == task_img_url:
                        match = v
                        break
                if not match:
                    log(f"No task match for image url {task_img_url}")
                    skipped += 1
                    continue
                task_id, _ = match
            # post prediction
            try:
                ls.predictions.create(
                    task=task_id,
                    model_version=args.model_version,
                    score=avg_score,
                    result=results
                )
                posted += 1
            except Exception as e:
                log(f"Failed to post prediction for task {task_id}: {e}")
                skipped += 1
                continue
            continue  # skip to next image

        # --- Call DINO-X API ---
        try:
            dinox = call_dinox_api(
                token=args.token,
                image_path=img_path,
                mask_format=args.mask_format,
                bbox_threshold=args.bbox_threshold,
                iou_threshold=args.iou_threshold
            )
        except Exception as e:
            log(f"API error for {img_path}: {e}")
            skipped += 1
            continue

        # map labels
        dinox = apply_label_mapping(dinox, MAPPING)

        # image size
        try:
            with PILImage.open(img_path) as im: W,H = im.size
        except Exception as e:
            log(f"Cannot read image size for {img_path}: {e}")
            skipped += 1
            continue

        # convert to LS results
        results, scores = build_results(dinox, W, H, rect_from, brush_from, to_name, label_map)
        if not results:
            log(f"No results for {img_path}")
            skipped += 1
            continue
        avg_score = float(np.mean(scores)) if scores else 0.0

        # match task by filename
        key = norm_filename(img_path)
        match = task_map.get(key)
        if not match:
            candidates = [(k, v) for k, v in task_map.items() if key in k]
            if len(candidates) == 1:
                match = candidates[0][1]
                log(f"Fuzzy match: '{key}' found in '{candidates[0][0]}'")
            elif len(candidates) > 1:
                match = candidates[0][1]
                log(f"Multiple fuzzy matches for '{key}': {[c[0] for c in candidates]}, using first.")
            else:
                log(f"No task match for {img_path} (normalized '{key}')")
                skipped += 1
                continue
        task_id, task_img_url = match

        # post prediction
        try:
            ls.predictions.create(
                task=task_id,
                model_version=args.model_version,
                score=avg_score,
                result=results
            )
            posted += 1
        except Exception as e:
            log(f"Failed to post prediction for task {task_id}: {e}")
            skipped += 1
            continue

        # optional write JSON (always use normalized filename)
        if args.output_dir:
            base = os.path.splitext(norm_base)[0] + ".json"
            outp = os.path.join(args.output_dir, base)
            with open(outp, "w", encoding="utf-8") as f:
                json.dump({
                    "data": {data_key: task_img_url},
                    "predictions": [{
                        "model_version": args.model_version,
                        "score": avg_score,
                        "result": results
                    }]
                }, f, ensure_ascii=False, indent=2)
            wrote += 1

    if args.output_dir:
        log(f"Wrote {wrote} JSON files to {args.output_dir}")
    log(f"Posted {posted} predictions. Skipped {skipped}.")

    if unmatched_labels:
        log("Unmatched label counts:")
        for label, count in unmatched_labels.items():
            log(f"{label} - {count}")
        log(f"All unmatched - {total_unmatched}")

        # --- Append/update unmatched label counts to CSV ---
        import csv
        csv_path = args.unmatched_csv
        # Read existing counts if file exists
        existing = {}
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline='', encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        label, count = row[0], row[1]
                        try:
                            existing[label] = int(count)
                        except Exception:
                            continue
        # Update with new counts
        for label, count in unmatched_labels.items():
            existing[label] = existing.get(label, 0) + count
        # Write back to CSV (overwrite)
        with open(csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            for label, count in existing.items():
                writer.writerow([label, count])
        log(f"Unmatched label counts written/appended to {csv_path}")

if __name__ == "__main__":
    main()
