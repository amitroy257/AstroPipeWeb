from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_image_folder_dataset(root_dir: str, img_size: int):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root.resolve()}")

    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(
            f"No class subfolders found in {root.resolve()}.\n"
            "Expected: dataset/classA/*.jpg, dataset/classB/*.jpg ..."
        )

    class_names = sorted([p.name for p in class_dirs])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    X_list, y_list = [], []
    total = 0

    for c in class_names:
        paths = [p for p in (root / c).rglob("*") if p.suffix.lower() in IMG_EXTS]
        total += len(paths)

    if total == 0:
        raise ValueError(f"No images found under {root.resolve()}")

    for c in class_names:
        paths = [p for p in (root / c).rglob("*") if p.suffix.lower() in IMG_EXTS]
        for img_path in tqdm(paths, desc=f"Loading {c}", leave=False):
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            X_list.append(arr.reshape(-1))  # flatten
            y_list.append(class_to_idx[c])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y, class_names