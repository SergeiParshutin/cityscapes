import argparse, yaml, torch, torch.nn as nn
from torch.optim import AdamW
import segmentation_models_pytorch as smp
from src.dataset import build_loaders
from src.utils import save_ckpt
import numpy as np

def miou_from_confmat(cm):
    # cm: CxC
    eps = 1e-7
    ious = []
    for i in range(cm.shape[0]):
        tp = cm[i,i]
        fn = cm[i,:].sum() - tp
        fp = cm[:,i].sum() - tp
        denom = tp + fp + fn + eps
        ious.append(tp / denom)
    return float(np.mean(ious))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f: cfg = yaml.safe_load(f)
    if args.epochs: cfg["epochs"] = args.epochs

    ignore_index = int(cfg.get("ignore_index", 255))
    num_classes = int(cfg.get("num_classes", 19))

    dl_tr, dl_va = build_loaders(cfg["data_root"], cfg["img_size"], cfg["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = getattr(smp, cfg.get("arch","Unet"))
    model = model_class(encoder_name=cfg["encoder_name"], encoder_weights=cfg["encoder_weights"],
                        classes=num_classes, activation=None).to(device)

    ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_miou = 0.0
    for ep in range(cfg["epochs"]):
        model.train()
        for x,y in dl_tr:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

        # val mIoU
        model.eval()
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        with torch.no_grad():
            for x,y in dl_va:
                x = x.to(device); y = y.numpy()
                logits = model(x).cpu()
                pred = logits.argmax(1).numpy()
                for pi, yi in zip(pred, y):
                    mask = yi != ignore_index
                    pi = pi[mask]; yi = yi[mask]
                    for c in range(num_classes):
                        for c2 in range(num_classes):
                            conf[c,c2] += int(((yi==c) & (pi==c2)).sum())
        miou = miou_from_confmat(conf) if conf.sum()>0 else 0.0
        print(f"Epoch {ep+1}/{cfg['epochs']}  mIoU={miou:.4f}")
        if miou > best_miou:
            best_miou = miou
            save_ckpt(model, f"{cfg['out_dir']}/ckpt.pt")

if __name__ == "__main__":
    main()
