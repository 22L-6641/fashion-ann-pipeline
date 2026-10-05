from pathlib import Path
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
np.savez_compressed(out / "train.npz", images=x_train, labels=y_train)
np.savez_compressed(out / "test.npz", images=x_test, labels=y_test)

print(f"Saved raw data: train={x_train.shape}, test={x_test.shape}")
