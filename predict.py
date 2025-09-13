from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from utils import get_cifar10_data

# Load data dan model
(_, _), (x_test, y_test) = get_cifar10_data()
model = load_model('checkpoints/best_model.h5')

# Prediksi 25 gambar pertama
probs = model.predict(x_test[:25])
preds = np.argmax(probs, axis=-1)

# Tampilkan hasil
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Pred: {preds[i]} | True: {y_test[i]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()