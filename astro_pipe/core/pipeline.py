from pathlib import Path
import json
from datetime import datetime
import numpy as np

from sklearn.model_selection import train_test_split

from astro_pipe.utils.config import load_config
from astro_pipe.data.image_folder_loader import load_image_folder_dataset
from astro_pipe.models.baseline import build_model
from astro_pipe.evaluation.metrics import evaluate


def run_pipeline(config_path: str) -> None:
    # ----------------------------
    # 1. Load configuration
    # ----------------------------
    cfg = load_config(config_path)

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    ds = cfg.get("dataset", {})
    root_dir = ds.get("root_dir", "dataset")
    img_size = int(ds.get("img_size", 64))
    test_size = float(ds.get("test_size", 0.2))

    out_cfg = cfg.get("output", {})
    runs_dir = Path(out_cfg.get("runs_dir", "runs"))
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 2. Load dataset
    # ----------------------------
    X, y, class_names = load_image_folder_dataset(root_dir, img_size)

    # ----------------------------
    # 3. Train/Test split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # ----------------------------
    # 4. Balance training set ONLY
    # ----------------------------
    class_counts = np.bincount(y_train)
    min_count = np.min(class_counts)

    balanced_indices = []

    for cls in range(len(class_counts)):
        cls_indices = np.where(y_train == cls)[0]

        if len(cls_indices) > min_count:
            cls_indices = np.random.choice(
                cls_indices,
                size=min_count,
                replace=False
            )

        balanced_indices.extend(cls_indices)

    balanced_indices = np.array(balanced_indices)

    X_train = X_train[balanced_indices]
    y_train = y_train[balanced_indices]

    # ----------------------------
    # 5. Train model
    # ----------------------------
    model = build_model(cfg.get("model", {}))
    model.fit(X_train, y_train)

    # ----------------------------
    # 6. Evaluate
    # ----------------------------
    y_pred = model.predict(X_test)
    results = evaluate(y_test, y_pred, class_names)

    # ----------------------------
    # 7. Save outputs
    # ----------------------------
    (run_dir / "config.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8"
    )

    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": results["accuracy"],
                "confusion_matrix": results["confusion_matrix"],
                "class_names": class_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (run_dir / "report.txt").write_text(
        results["report"],
        encoding="utf-8"
    )

    # ----------------------------
    # 8. Print results
    # ----------------------------
    print(f"[OK] Run saved to: {run_dir.resolve()}")
    print(f"[OK] Accuracy: {results['accuracy']:.4f}")
    print()
    print(results["report"])