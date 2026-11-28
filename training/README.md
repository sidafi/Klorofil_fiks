# Training: Model Jenis dan Kesehatan Daun

Folder ini berisi semua yang dibutuhkan untuk membangun ulang data, melatih, dan mengevaluasi dua model yang dipakai web app Klorofil.
Hasil akhirnya ada di `model/` di root repo dan dibaca oleh `predictor.py`.

| Model | Tugas | Metode | Berkas |
|---|---|---|---|
| Kesehatan | daun `sakit` / `sehat` | CNN dibangun dari nol | `model/kesehatan.tflite`, `model/kesehatan_labels.json` |
| Jenis | 9 kelas jenis tanaman | Transfer learning MobileNetV2 (ImageNet) | `model/jenis.tflite`, `model/jenis_labels.json` |

## Isi folder

```
training/
├── persiapan_data.ipynb      # bangun ulang data + bundle zip dari dataset asli (dijalankan lokal)
├── training_kesehatan.ipynb  # training + evaluasi + ekspor model kesehatan (Google Colab, GPU T4)
├── training_jenis.ipynb      # training + evaluasi + ekspor model jenis (Google Colab, GPU T4)
├── manifest/
│   ├── data_kesehatan.csv        # daftar pasti 44.739 citra: file, split, label, source, asal
│   ├── data_jenis.csv            # daftar pasti 12.746 citra
│   └── atribusi_inaturalist.csv  # atribusi foto iNaturalist (lisensi CC)
└── hasil/
    ├── kesehatan/   # history.csv/.png, confusion matrix val & test, report_test.txt
    └── jenis/
```

## Alur kerja

1. **Unduh dataset** ke satu folder induk (lihat tabel Dataset).
2. **`persiapan_data.ipynb`** (lokal): membaca manifest, memproses citra, membuat augmentasi ganti latar, lalu menghasilkan `colab_bundle.zip` (kesehatan) dan `jenis_bundle.zip` (jenis).
3. Upload kedua zip ke `MyDrive/` Google Drive.
4. Jalankan **`training_kesehatan.ipynb`** dan **`training_jenis.ipynb`** di Colab. Keduanya menyimpan model terbaik, history, confusion matrix, lalu mengekspor `.tflite` + `labels.json`.
5. Salin `.tflite` dan `labels.json` ke `model/`.

Kebutuhan lokal untuk `persiapan_data.ipynb`: Python 3.9+, `pandas`, `numpy`, `pillow`, `opencv-python`, `matplotlib`.
Notebook training memakai TensorFlow/Keras bawaan Colab dan `scikit-learn`.

## Dataset

| Folder (di folder induk) | Dataset | Dipakai untuk |
|---|---|---|
| `archive (1)/PlantVillage/` | PlantVillage (Kaggle) | kesehatan, jenis |
| `archive (2)/leaves_healthy_or_diseased/` | Leaves healthy or diseased (Kaggle) | kesehatan, jenis |
| `archive/rice_leaf_diseases/` | Rice leaf diseases (Kaggle) | kesehatan, jenis |
| `archive (3)/PlantDoc/` | PlantDoc (datasetninja) | kesehatan, jenis |
| `archive (4)/inaturalist/` | iNaturalist, kelompok penyakit/sehat | kesehatan, jenis |
| `archive (5)/inat_jenis/` | iNaturalist, per jenis tanaman | jenis |

Dataset mentah dan zip tidak di-commit (`.gitignore`). Yang di-commit hanya **manifest**, yaitu daftar pasti setiap citra beserta split dan label, sehingga data bisa dibangun ulang persis sama.

### Data kesehatan: 44.739 citra

| Split | Jumlah | Catatan |
|---|---|---|
| train | 33.323 | 26.633 asli + 6.690 augmentasi ganti latar (`bgaug__*`) |
| val | 5.708 | |
| test | 5.708 | |

Split 70/15/15 bertingkat per label × sumber, dibuat **sebelum** augmentasi.

### Data jenis: 12.746 citra, 9 kelas

`jagung, jambu_biji, kentang, lainnya, mangga, monstera, padi, paprika_cabai, tomat` (urutan = indeks output model).
Kelas `lainnya` berisi daun tanaman di luar 8 jenis. Split: train 8.929, val 1.893, test 1.924.
Kuota tiap kelas diisi dengan foto lapangan terlebih dulu, baru foto studio, supaya model tidak didominasi foto berlatar polos.

## Preprocessing

Wajib identik di training dan `predictor.py`:
RGB → crop persegi di tengah → resize 128×128 (LANCZOS) → bagi 255.
Model jenis menskalakan ulang input ke [-1, 1] **di dalam model** (layer `Rescaling`), jadi backend tetap mengirim input 0–1.

Augmentasi saat training (hanya train): flip horizontal & vertikal, rotasi 0,1, zoom 0,1, kontras 0,1.

**Augmentasi ganti latar** (data kesehatan, train): daun dari foto berlatar polos disegmentasi lewat HSV (saturasi > 45, kecerahan > 40, buang hue biru, morfologi, komponen terbesar, isi lubang), lalu ditempel ke latar baru: 70% potongan foto lapangan yang di-blur, 30% gradien hijau/tanah. Tujuannya agar model tidak memakai latar sebagai jalan pintas.

## Arsitektur dan training

**Kesehatan (CNN dari nol, 423.010 parameter).**
4 blok `[Conv 3×3 (32/64/128/256) → BatchNorm → ReLU → MaxPool]` → GlobalAveragePooling → Dropout 0,4 → Dense 128 ReLU → Dense 2 softmax.
Adam lr 1e-3, tanpa `class_weight`, EarlyStopping (patience 12), ReduceLROnPlateau (×0,5, patience 3). Epoch terbaik 16 dari 28.

**Jenis (transfer learning, 2.269.513 parameter).**
MobileNetV2 ImageNet (input 128) → GlobalAveragePooling → Dropout 0,3 → Dense 9 softmax.
Fase 1: base dibekukan, 15 epoch, lr 1e-3. Fase 2: 40 layer teratas dibuka (BatchNorm tetap beku), 20 epoch, lr 1e-5.

## Hasil

Aturan penerimaan dikunci **sebelum** training: satu run, test dijalankan sekali dan hanya jika val lolos.

| Model | Syarat (macro F1) | Val | Test | Akurasi test | Status |
|---|---|---|---|---|---|
| Kesehatan | val ≥ 0,885, test ≥ 0,873 | 0,8913 | **0,8867** | 0,904 | Lolos |
| Jenis | val ≥ 0,793, test ≥ 0,784 | 0,8199 | **0,8077** | 0,797 | Lolos |

**Kesehatan.** Recall `sakit` 0,893, precision `sehat` 0,768, jadi model sedikit condong menebak `sehat`.
Macro F1 test per sumber: PlantVillage 0,966, leaves 0,950, iNaturalist 0,800, PlantDoc 0,738.

**Jenis.** Kelas terlemah: `lainnya` (F1 0,63) dan `tomat` (0,77). Foto studio hampir sempurna (±0,98), foto lapangan 0,64–0,78.

Rincian lengkap: `hasil/<model>/report_test.txt` dan confusion matrix di folder yang sama.

### Ambang "tidak yakin"

Dipilih di data **val** (bukan test): ambang terkecil yang membuat akurasi prediksi yakin ≥ 0,95.

| Model | Ambang | Porsi val yang yakin | Akurasi saat yakin |
|---|---|---|---|
| Kesehatan | **0,80** | 82,3% | 0,964 |
| Jenis | **0,85** | 61,6% | 0,952 |

## Kontrak dengan backend

- `labels.json` = urutan kelas sesuai indeks output softmax: kesehatan `["sakit", "sehat"]`, jenis seperti daftar di atas.
- Kedua model menerima input `float32` berbentuk `(1, 128, 128, 3)` dengan nilai 0–1 dan mengeluarkan probabilitas softmax.
- Ambang yang dipakai `predictor.py`: `THRESHOLD_JENIS = 0.85` dan `THRESHOLD_KESEHATAN = 0.80` (hasil tuning di val pada tabel di atas).

## Keterbatasan

- Model masih sebagian belajar dari **latar dan sumber dataset**: foto lapangan jauh lebih sulit daripada foto studio. Metrik selalu dilaporkan per sumber.
- Segmentasi HSV membuang bagian daun yang pucat: 681 dari 6.690 citra augmentasi (±10%) luas daunnya < 15%, sebagian hanya berisi potongan daun.
- Augmentasi ganti latar yang dibangun ulang tidak identik byte per byte dengan data yang dipakai training (urutan angka acak berbeda); val dan test identik.
- `predictor.py` tidak memakai `draft` JPEG seperti saat praproses data, sehingga foto JPEG besar bisa menghasilkan piksel sedikit berbeda.
