from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

test = np.load("data/processed/test.npz")
model = load_model("models/model.h5")
loss, accuracy = model.evaluate(test["images"], test["labels"], verbose=0)
pred = model.predict(test["images"], verbose=0).argmax(axis=1)

metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy)}
with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

Path("reports").mkdir(exist_ok=True)
cm = confusion_matrix(test["labels"], pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation="vertical")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
plt.close()

print(json.dumps(metrics, indent=2))
