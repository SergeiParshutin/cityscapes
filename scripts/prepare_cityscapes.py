import argparse, shutil
from pathlib import Path
import cv2
import numpy as np

# Cityscapes labelIds -> trainIds mapping (subset to 19 classes; others -> 255)
# Reference: official mapping
LABELID_TO_TRAINID = {
    7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9,
    23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18
}
IGNORE_ID = 255

def convert_labelIds_to_trainIds(label_img):
    out = np.full_like(label_img, IGNORE_ID)
    for lid, tid in LABELID_TO_TRAINID.items():
        out[label_img == lid] = tid
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to Cityscapes root with leftImg8bit/ and gtFine/")
    ap.add_argument("--out", required=True, help="Output data root (will create images/* and masks/*)")
    ap.add_argument("--split", choices=["trainval","train","val"], default="trainval", help="Which splits to prepare")
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    img_root = raw / "leftImg8bit"
    gt_root  = raw / "gtFine"
    assert img_root.exists() and gt_root.exists(), "Expect leftImg8bit/ and gtFine/ in --raw"

    splits = ["train","val"] if args.split == "trainval" else [args.split]
    for split in splits:
        src_imgs = img_root / split
        src_masks = gt_root / split
        dst_img_split = out / "images" / split
        dst_mask_split = out / "masks" / split
        dst_img_split.mkdir(parents=True, exist_ok=True)
        dst_mask_split.mkdir(parents=True, exist_ok=True)

        for city_dir in sorted(src_imgs.iterdir()):
            if not city_dir.is_dir(): continue
            for img_path in city_dir.glob("*_leftImg8bit.png"):
                # Corresponding gt file
                mask_path = src_masks / city_dir.name / img_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                if not mask_path.exists():
                    print(f"WARNING: missing mask for {img_path.name}")
                    continue
                # Copy image
                shutil.copy2(img_path, dst_img_split / img_path.name)
                # Convert mask
                label = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                train_ids = convert_labelIds_to_trainIds(label)
                cv2.imwrite(str(dst_mask_split / img_path.name), train_ids)
        print(f"Prepared split: {split}")

if __name__ == "__main__":
    main()
