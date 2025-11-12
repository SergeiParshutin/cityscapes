# Datu ielādēšanas palīgfunkcijas
# Importējam bibliotēkas
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A          # Bibliotēka ar standartizētām datu pieaudzināšanas (augmentation) funkcijām
from albumentations.pytorch import ToTensorV2

IGNORE_INDEX = 255
NUM_CLASSES = 19  # Cityscapes trainIds: 0..18, 255=ignore

class CityscapesDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size_h=96, img_size_w=256):
        self.imgs = sorted(list(Path(img_dir).rglob('*.png')) + list(Path(img_dir).rglob('*.jpg')))
        self.mask_dir = Path(mask_dir)
        self.size = (img_size_h, img_size_w)
        self.tfms = A.Compose([
            A.Resize(img_size_h, img_size_w, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=IGNORE_INDEX),
            A.Normalize(),
            ToTensorV2()
        ])
        self.tfms_val = A.Compose([
            A.Resize(img_size_h, img_size_w, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(),
            ToTensorV2()
        ])
        self.training = True  # tiks pārrakstīts builderī

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.mask_dir / img_path.name
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Vērtības gaidāmas 0..18 vai 255

        if mask is None:
            raise FileNotFoundError(f"Mask not found for {img_path.name}")

        # Sanitizācija: kaut kas ārpus 0..18 kļūst par ignore (255)
        # (piem., ja maska nav trainIds vai resizing radījis artefaktus)
        mask = mask.astype(np.int16)
        bad = (mask != IGNORE_INDEX) & ((mask < 0) | (mask > (NUM_CLASSES-1)))
        if bad.any():
            mask[bad] = IGNORE_INDEX
        mask = mask.astype(np.uint8)

        # Albumentations (resize ar NEAREST maskai)
        if self.training:
            aug = self.tfms(image=img, mask=mask)
        else:
            aug = self.tfms_val(image=img, mask=mask)

        x, y = aug["image"], aug["mask"].long()  # y: (H,W) long (0..18, 255)

        # Papildu drošība: ja tomēr kaut kas ārpus diapazona, aizvieto ar IGNORE
        if torch.any((y != IGNORE_INDEX) & ((y < 0) | (y >= NUM_CLASSES))):
            y = torch.where((y < 0) | (y >= NUM_CLASSES), torch.tensor(IGNORE_INDEX, dtype=y.dtype), y)

        return x, y

def build_loaders(root, img_size_h=96, img_size_w=256, batch_size=4, num_workers=0):
    # Piezīme Windows lietotājiem: vispirms var būt drošāk ar num_workers=0
    tr = CityscapesDataset(f"{root}/images/train", f"{root}/masks/train", img_size_h, img_size_w)
    va = CityscapesDataset(f"{root}/images/val",   f"{root}/masks/val",  img_size_h, img_size_w)
    tr.training = True; va.training = False
    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False)
    dl_va = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return dl_tr, dl_va
