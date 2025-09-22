# DINO-X Integration with Label Studio
Load DINO-X annotations via API calls as pre-annotations to Label Studio


## Prerequisites
- Python 3.9–3.12.
- Label Studio Access Token
- DINO-X API token
- Your Label Studio project must include `RectangleLabels` for boxes and `BrushLabels` for masks. Configure these in **Settings → Labeling Interface**.

## Install
(Conda Environment recommended)

```bash
conda create -n env
conda activate env
pip install -r requirements.txt
