# Cityscapes — Semantic Segmentation (UNet / PyTorch)

Multi-class semantic segmentation with 19 **train IDs** on the Cityscapes dataset
(urban street scenes). Uses `segmentation-models-pytorch` (SMP) for quick fine-tuning.

## Dataset setup

Cityscapes requires registration and manual download.
Download the following archives **to `data/raw/`** and extract:

- `leftImg8bit_trainvaltest.zip`
- `gtFine_trainvaltest.zip`

After extraction you should have:
```
data/raw/cityscapes/
  leftImg8bit/ train/ val/ test/ .../*.png
  gtFine/      train/ val/ test/ .../*_gtFine_labelIds.png
```

Then **prepare train/val** structure and convert labelIds -> 19-class `trainIds` (0..18, ignore=255):
```bash
python scripts/prepare_cityscapes.py --raw data/raw/cityscapes --out data
```

This will create:
```
data/
  images/train/*.png
  images/val/*.png
  masks/train/*.png    # values in {0..18, 255}
  masks/val/*.png
```

## Quickstart (training)

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/config.yaml
python scripts/eval.py --weights runs/ckpt.pt --data-root data --img-size 512
```

## Streamlit demo
```bash
streamlit run app.py
```

## Docker quickstart
```bash
make docker-build
make docker-run   # open http://localhost:8501
```
