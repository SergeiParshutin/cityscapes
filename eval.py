# Satur funkcijas modeļa novērtēšanai

# Importējam bibliotēkas
import argparse, torch, numpy as np
import segmentation_models_pytorch as smp
from src.dataset import build_loaders

def miou_from_confmat(cm):
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
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--img_size_h", type=int, default=96)
    ap.add_argument("--img_size_w", type=int, default=256)
    ap.add_argument("--encoder", type=str, default="resnet34")
    ap.add_argument("--arch", type=str, default="Unet")
    ap.add_argument("--num-classes", type=int, default=19)
    ap.add_argument("--ignore-index", type=int, default=255)
    args = ap.parse_args()

    dl_tr, dl_va = build_loaders(args.data_root, args.img_size_h, args.img_size_w, 4)
    model_class = getattr(smp, args.arch)
    model = model_class(encoder_name=args.encoder, encoder_weights=None, classes=args.num_classes, activation=None)
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()

    conf = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    with torch.no_grad():
        for x,y in dl_va:
            logits = model(x)
            pred = logits.argmax(1).numpy()
            y = y.numpy()
            mask = y != args.ignore_index
            for pi, yi in zip(pred, y):
                pi = pi[mask[0]]  # careful: mask shape equals yi shape; handle per-sample mask
            # Simpler per-sample loop:
    # Recompute per-sample to avoid masking confusion
    conf = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    with torch.no_grad():
        for x,y in dl_va:
            logits = model(x).cpu()
            pred = logits.argmax(1).numpy()
            y = y.numpy()
            for pi, yi in zip(pred, y):
                m = yi != args.ignore_index
                pi = pi[m]; yi = yi[m]
                for c in range(args.num_classes):
                    for c2 in range(args.num_classes):
                        conf[c,c2] += int(((yi==c) & (pi==c2)).sum())
    print(f"Val mIoU={miou_from_confmat(conf):.4f}")

if __name__ == "__main__":
    main()
