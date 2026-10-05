from pathlib import Path
import csv
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

with open("params.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["train"]

train = np.load("data/processed/train.npz")
val = np.load("data/processed/val.npz")

model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(int(cfg["dense_units"]), activation="relu"),
    Dropout(float(cfg["dropout_rate"])),
    Dense(10, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["learning_rate"])),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train["images"], train["labels"],
    validation_data=(val["images"], val["labels"]),
    epochs=int(cfg["epochs"]),
    batch_size=int(cfg["batch_size"]),
    verbose=2,
)

out = Path("models")
out.mkdir(parents=True, exist_ok=True)
model.save(out / "model.h5")

with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    keys = list(history.history.keys())
    writer.writerow(keys)
    writer.writerows(zip(*(history.history[k] for k in keys)))

print("Saved models/model.h5 and models/history.csv")
