# Cityscapes — Semantic Segmentation (UNet / PyTorch)

Daudzklašu semantiskā segmentācija ar 19 klasēm un Unet tīklu, izmantojot Cityscapes datu kopas daļu.
Izmanto `segmentation-models-pytorch` (SMP) ātrakam fine-tuning procesam.

## Datu kopas sagatavošana
Datu kopai ir jābūt organizētai šādi:
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
python scripts/eval.py --weights runs/ckpt.pt --data-root data --img_size_h 96 --img_size_w 256
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

DITEF klasterī tiek lietots CUDA 13.0, tāpēc pytorch ir jāinstalē ar
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
