from pathlib import Path
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["preprocess"]

raw = Path("data/raw")
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

train = np.load(raw / "train.npz")
test = np.load(raw / "test.npz")

x = train["images"].astype("float32") / 255.0
y = train["labels"]
x_test = test["images"].astype("float32") / 255.0
y_test = test["labels"]

x_train, x_val, y_train, y_val = train_test_split(
    x, y,
    test_size=cfg["test_size"],
    random_state=cfg["seed"],
    stratify=y
)

np.savez_compressed(out / "train.npz", images=x_train, labels=y_train)
np.savez_compressed(out / "val.npz", images=x_val, labels=y_val)
np.savez_compressed(out / "test.npz", images=x_test, labels=y_test)

print(f"Saved processed data: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")
