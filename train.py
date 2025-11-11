# Importējam bibliotēkas
import argparse, yaml, torch, torch.nn as nn
from torch.optim import AdamW
import segmentation_models_pytorch as smp
from src.dataset import build_loaders
from src.utils import save_ckpt, get_device
import numpy as np
from tqdm.auto import tqdm

# Funkcija vidējās IoU aprēķināšanai - mIoU, no pārpratumu matricas
# Katrai klasei tiek aprēķināta IoU un tad tiek aprēķināta visu IoU vidējā vērtība
def miou_from_confmat(cm):
    # cm: CxC
    eps = 1e-7
    ious = []
    for i in range(cm.shape[0]):
        tp = cm[i,i]                    # Īstie pozitīvie - pa diagonāli
        fn = cm[i,:].sum() - tp         # Kļūdainie negatīvie
        fp = cm[:,i].sum() - tp         # Kļūdainie pozitīvie
        denom = tp + fp + fn + eps      # Saucējs IoU aprēķinam. Eps pievienota, lai nebūtu dalīšanai ar 0, ja tp=fp=fn=0
        ious.append(tp / denom)         # Pievienojam IoU kārtējai klasei
    return float(np.mean(ious))         # Atgriežam vidējo IoU - mIoU

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
    print(f'Aprēķini tiek veikti ar iekārtu: {device}')
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
    for ep in range(cfg["epochs"]): # Cikls pa epohām
        model.train()               # Pārslēdzam modeli uz apmācības režimu
        # Parametri priekš progress-bar
        running_loss = 0.0
        pbar = tqdm(dl_tr, total=len(dl_tr),
                desc=f"Epoch {ep+1}/{cfg['epochs']} [train]", unit="batch")
        for i, (x, y) in enumerate(pbar, 1):    # Katram ierakstu pārim - batcham
            x,y = x.to(device), y.to(device)    # Augšupielādējam uz iekārtu, jo datiem ir jābūt vienā iekārtā ar modeli
            logits = model(x)       # Iegustam prognozi
            loss = ce(logits, y)    # Apēķinam loss vērtību
            opt.zero_grad()         # Nometam gradientu pirms aprēķināt jaunu
            loss.backward()         # Izplatam kļūdu atpakaļ pa slāņiem
            opt.step()              # Signalizējam, ka nosacītais swaru pielāgošanas solis ir pabeigts
            
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss/i:.4f}")

        # val mIoU
        model.eval()                # Epohas beigās pārsledzam modeli uz novērtēšanas (evaluation) režīmu
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)     # Inicializējam pārpratumu matricu
        with torch.no_grad():       # Izslēdzam gradientu, lai svari netiktu mainīti
            pbar_val = tqdm(dl_va, total=len(dl_va),
                        desc=f"Epoch {ep+1}/{cfg['epochs']} [val]", unit="batch")
            for x, y in pbar_val:       # Pa batchiem validācijas kopā
                x = x.to(device)        # Ieejas datus sūtam uz iekārtu, jo tur ir modelis. 
                y = y.numpy() 
                logits = model(x)       # Prognozējam klašu varbūtības katram pikselim
                pred = logits.argmax(1).cpu().numpy()     # Izvēlamies klasi ar lielāko varbūtību. NB! Vispirms atgrižeam tenzoru uz CPU.
                for pi, yi in zip(pred, y):     # Katram piklselim reālajā maskā un prognozētajā maskā
                    mask = yi != ignore_index   # Aprēķinam maskas korekciju, izslēdzot ignore_index vērtības (255)
                    pi = pi[mask]               # Koriģējam prognozēto masku
                    yi = yi[mask]               # Koriģējam reālo masku
                    for c in range(num_classes):        # Aizpildam pārpratumu matricu
                        for c2 in range(num_classes):
                            conf[c,c2] += int(((yi==c) & (pi==c2)).sum())
        miou = miou_from_confmat(conf) if conf.sum()>0 else 0.0     # Aprēķinam mIoU
        print(f"Epoch {ep+1}/{cfg['epochs']}  mIoU={miou:.4f}")     # Paziņojam rezultātu
        if miou > best_miou:        # Ja kārtējais mIoU ir labāks pār tiklīdz labako mIoU, tad atjaunojam best_miou vērtību un saglabājam modeli
            best_miou = miou
            save_ckpt(model, f"{cfg['out_dir']}/ckpt.pt")

if __name__ == "__main__":
    main()
