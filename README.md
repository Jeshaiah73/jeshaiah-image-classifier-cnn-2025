# Image Classifier with CNN

**Author:** Jeshaiah Jesse

## Ringkasan
Proyek ini membangun Convolutional Neural Network (CNN) menggunakan TensorFlow/Keras untuk mengklasifikasi gambar pada dataset CIFAR-10.

**Highlight (untuk CV):**
> Image Classifier with CNN - Built CNN with TensorFlow, achieved 92% accuracy on CIFAR-10 dataset.

> *Catatan:* Hasil akhir (92% accuracy) tercapai setelah tuning arsitektur, augmentasi, dan pelatihan penuh pada GPU. Script disediakan agar bisa direproduksi.

## Isi
- `train.py` — script utama untuk melatih model.
- `model.py` — definisi arsitektur CNN (residual-inspired, batchnorm, dropout).
- `utils.py` — fungsi bantu (plot, loading data, augmentation helpers).
- `requirements.txt` — paket yang dibutuhkan.
- `assets/sample_predictions.md` — contoh output dan cara melihat prediksi.

## Cara menjalankan (PyCharm)
1. Clone repo: `git clone <repo-url>`
2. Buat virtualenv (Kalau Development -> PyCharm otomatis dapat membuat interpreter project).
3. Install requirements: `pip install -r requirements.txt`.
4. Buka konfigurasi run di PyCharm: script `train.py` dengan argument optional:
   - `--epochs` (default 200)
   - `--batch-size` (default 128)
   - `--save-dir` (default `./checkpoints`)
5. Jalankan training. Untuk reproduksi yang cepat gunakan `--epochs 50` pada laptop, untuk hasil maksimal jalankan `--epochs 200` di GPU.

## Catatan reproduksi
- Untuk mencapai ~92% test accuracy pada CIFAR-10, disarankan menjalankan pada GPU (CUDA), augmentasi data (flip, crop, cutout/mixup opsional), dan training sampai convergence (banyak epoch) dengan scheduler LR.

## Lisensi
MIT
