# DINO-X Integration with Label Studio
Load DINO-X annotations via API calls as pre-annotations to Label Studio

## Prerequisites
- Python 3.9–3.12 (and requirements.txt - follow installation instructions below)
- [Label Studio Legacy Access Token](https://labelstud.io/guide/access_tokens)
- [DINO-X API token](https://github.com/deepdataspace/dds-cloudapi-sdk?tab=readme-ov-file#3-apply-for-an-api-token)
- Your Label Studio project must include `RectangleLabels` for boxes and `BrushLabels` for masks. Configure these in **Settings → Labeling Interface**.
- original filenames (or atleast similar) from the dataset are maintened, or the same files that have been uploaded to Label Studio are used for these scripts.
- Prelabelling enabled in Project->Settings->Annotation->Prelabelling with DINO-X.

## Install
(Conda Environment recommended)

```bash
conda create -n env
conda activate env
pip install -r requirements.txt
```

## Overview

Two ways to pre-annotate Label Studio with DINO-X:

<u>One-step</u>: dinox_to_LS.py runs DINO-X on images and immediately posts predictions to existing LS tasks. Optionally writes per-image JSONs (in **Label Studio format**).

<u>Two-step</u>:
* call_dinox_api.py calls DINO-X and saves **raw prediction** JSONs.
* post_predictions_to_LS.py converts those JSONs to LS predictions and posts them.

Both paths support bbox+mask, COCO RLE→LS RLE conversion, label normalization, task matching by filename, and per-image JSON export.

## Script A: dinox_to_LS.py (one-step)

Purpose: Run DINO-X on local images already imported as LS tasks, post predictions, optionally save one JSON per image.

**Key features**

* DINO-X call with prompt-free “universal” mode, bbox+mask targets, COCO RLE masks.

* Label mapping via a predefined dictionary (e.g., “car”→“civilian vehicle”).

* Converts COCO RLE to LS brush RLE and xyxy bbox to percent coords.

* Matches LS tasks by normalized filename (handles hashed prefixes, fuzzy fallback).

* Tracks unmatched labels and appends counts to CSV.

* Skip logic example: if a JSON exists and task already has predictions, it skips; if JSON exists but task has 0 predictions, it loads from the JSON instead of calling DINO-X.

**Arguments**

``--token`` DINO-X API token.

``--input_dir`` Directory of images already present as LS tasks.

``--bbox_threshold, --iou_threshold`` DINOX thresholds (set to 0.25 and 0.8 by default respectively).

``--api_key`` Label Studio Access Token.

``--project_id`` example: http://10.100.52.107:8080/projects/22/settings -> project_id = 22

``--output_dir`` Write a JSON per image with the posted prediction payload.

``--unmatched_csv`` File to maintain unmatched label counts (by default set to ./unmatched_csv)

<u>Usage</u>
```bash
python dinox_to_LS.py \
  --token $DINOX_TOKEN \
  --input_dir /path/to/images \
  --api_key $LS_API_KEY \
  --project_id 18 \
  --output_dir ./jsons
```

Result: predictions posted to matching tasks, JSONs stored in jsons/

## Script B1: call_dinox_api.py (two-step: step 1)

Purpose: Batch call DINO-X and save raw JSONs next to images. GUI file picker by default.

**Key features**

* Uses DINO-X v2/task/dinox/detection with prompt type “universal,” bbox+mask, thresholds configurable in code.

* Optional COCO RLE decoding to binary masks; off by default.

* Applies the same label mapping table before saving.

**Configuration (edit file)**

``MY_TOKEN``: set to your token.

``DECODE_RLE``: False recommended for the next step to keep COCO RLE.

<u>Usage</u>

```python
python call_dinox_api.py
# choose images in the dialog; it writes <image>_dinox_preannot.json
```

Output: per-image *_dinox_preannot.json with "objects" list.

## Script B2: post_predictions_to_LS.py (two-step: step 2)
Purpose: Read DINO-X JSONs, convert to LS prediction result, and post to existing tasks. Can also write an LS-shaped JSON per image.

Similar usage to Script A

* ```--input_dir``` Directory containing DINO-X JSONs from script B1.

<u>Usage</u>

```bash
python post_predictions_to_LS.py \
  --input_dir ./preannots \
  --api_key $LS_API_KEY \
  --project_id 18 \
  --ls_url http://<host>:8080 \
  --output_dir ./ls_payloads
```
Result: predictions posted, optional LS-ready JSONs in ls_payloads/.