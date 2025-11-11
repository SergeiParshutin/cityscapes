# Importējam bibliotēkas
import argparse, yaml, torch, torch.nn as nn
from torch.optim import AdamW
import segmentation_models_pytorch as smp
from src.dataset import build_loaders
from src.utils import save_ckpt, get_device
import numpy as np

# Funkcija vidējās IoU aprēķināšanai - mIoU, no pārpratumu matricas
# Katrai klasei tiek aprēķināta IoU un tad tiek aprēķināta visu IoU vidējā vērtība
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

# Galvenā funkcija
def main():
    # Argumentu nolasīšana, ja tiks palaists caur komandrindu
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    # Nolasam config.yaml datni
    with open(args.config) as f: cfg = yaml.safe_load(f)
    # Ja epohu skaits ir norādīts parametros, tad ņemam to, nevis configā norādīto
    if args.epochs: cfg["epochs"] = args.epochs

    # Nolasam parametrus - indeksu, kuru ir jāignorē, un klašu skaitu, norādot noklusejuma vērtības, ja configā nav norādīts
    ignore_index = int(cfg.get("ignore_index", 255))
    num_classes = int(cfg.get("num_classes", 19))

    # Izveidojam DataLoaders, kuri ielādēs apmācības un validācijas kopas
    dl_tr, dl_va = build_loaders(cfg["data_root"], cfg["img_size_h"], cfg['img_size_w'], cfg["batch_size"])
    # Nosakam iekārtu, kurā notiks aprēķins. Prioritāte: cuda->mps->cpu
    device = get_device()
    # No torch modeļiem izveidojam modeļa klasi atbilstoši arhitektūrai - Unet
    model_class = getattr(smp, cfg.get("arch","Unet"))
    # Izveidojam modeli un sūtam to uz noteikto iekārtu
    model = model_class(encoder_name=cfg["encoder_name"], encoder_weights=cfg["encoder_weights"],
                        classes=num_classes, activation=None).to(device)
    # Definējam kļūdas (loss) funkciju
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    # Definējam apmācības koeficienta optimizatoru
    opt = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    best_miou = 0.0                 # Kļūdas sākotnējā vērtība - 0.0
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
