# Prediksi STR, stok, dan artikel serupa

Aplikasi Streamlit menggunakan `best_rf_mod = grid_rf.best_estimator_` dari GridSearchCV sesuai notebook terbaru: 324 kombinasi parameter, 5-fold CV, scoring R2, split 70/30 dan random_state 43. Model tetap fit pada X_train, bukan di-refit ke holdout. Encoder mengikuti notebook (fit pada df_reg eligible). Metrik per target disimpan di bundle.

## Perubahan perilaku

- Hanya artikel dengan tanggal launching <= tanggal proses dikurangi 90 hari masuk dataset. Tanggal launching mengikuti `MIN(stockawal.tanggal)` pada notebook; filter berlaku di SQL dan Python. Perhitungan tanggal menggunakan Asia/Bangkok (UTC+7, sama dengan WIB). Baris tanpa tanggal/artikel atau stok awal <=0/non-finite dikeluarkan. Artikel duplikat menyebabkan training gagal agar artikel holdout tidak bocor ke tabel kemiripan.
- Dropdown mengambil nilai unik df_reg eligible, null diganti 0. Bulan diurutkan numerik, ditampilkan Januari–Desember untuk bulan yang ada dalam data.
- Kemiripan = jumlah nilai sama pada sub_kategori, cutting, design_density, design_border, siluet, sleeve, collar, length_hem, color_theme / 9 × 100%. Semua atribut berbobot sama; nilai 0 vs 0 dihitung sama. Bulan tidak dihitung. Hanya >50% (minimal 5 dari 9) ditampilkan. Urutan utama tanggal launching menurun, lalu nama artikel untuk tanggal yang sama. Jumlah dihitung per artikel unik. Referensi hanya baris X_train milik model aktif, tidak termasuk X_test atau produk <90 hari.
- Rumus STR tetap mengikuti notebook: penjualan mulai 90 hari sebelum tanggal stok awal sampai hari ke-30/60/90 setelahnya, rasio dibatasi maksimal 1. Definisi ini tidak diubah menjadi jendela sejak launching. Prediksi stock tetap estimasi stok awal historis, bukan stok optimal.

## Aktivasi satu kali

1. Gabungkan perubahan ke `main`; Streamlit menjalankan `app.py` dengan Python 3.13 dan `requirements.txt` repository.
2. Gunakan bucket privat `klamby_sales` di Google Cloud. Bucket harus sudah ada; kode tidak membuat bucket atau mengubah izin publik.
3. Buat/pakai service account trainer. Beri `roles/bigquery.jobUser` di project `klamby-469003`, `roles/bigquery.dataViewer` di dataset sumber, dan `roles/storage.objectUser` di bucket `klamby_sales`. Jika tabel BigQuery bersumber Google Drive/Sheets, bagikan file sumber ke email service account dengan akses baca. Script meminta scope Drive seperti notebook.
4. Di repository GitHub: Settings > Secrets and variables > Actions > New repository secret. Isi `GCP_TRAINER_SA_JSON` dengan JSON service account trainer. Jangan commit JSON. Workflow memakai secret ini saat runtime. Untuk penguatan selanjutnya, auth dapat dimigrasikan ke Workload Identity Federation.
5. Buat service account pembaca dengan `roles/storage.objectViewer` hanya pada bucket. Masukkan JSON-nya ke Streamlit > App settings > Secrets sesuai `.streamlit/secrets.toml.example`; set `MODEL_BUCKET = "klamby_sales"`. Pembaca tidak perlu akses BigQuery.
6. Di GitHub Actions pilih **Train STR monthly > Run workflow** pada main untuk training pertama. Setelah sukses, aplikasi memuat model baru saat rerun/interaksi berikutnya (pemeriksaan manifest dicache 60 detik). Tidak perlu mengunggah model atau reboot per bulan.

Jadwal: tanggal 1 setiap bulan pukul 03.17 Asia/Bangkok/WIB. GitHub dapat menunda jadwal. **Pada repository publik, schedule dinonaktifkan GitHub setelah 60 hari tanpa aktivitas repository.** Pantau status Actions; untuk jadwal yang harus terus berjalan tanpa aktivitas repo, gunakan Cloud Scheduler + Cloud Run Job dengan perintah training yang sama. Tidak ada heartbeat atau perubahan commit buatan untuk menghindari aturan ini.

## Penyimpanan dan pergantian model

Bundle privat berada di `gs://klamby_sales/str-model/versions/<version>/model_bundle.joblib`; penanda aktif di `str-model/latest.json`. Bundle mencakup model, encoder, dropdown, artikel training, versi library, tanggal cutoff, dan metrik. Artikel/data training tidak di-commit ke repository publik atau diunggah sebagai artifact Actions publik.

Versi baru diaktifkan hanya setelah training, evaluasi finite, pemeriksaan prediksi dan roundtrip serialisasi berhasil. File diunggah terlebih dahulu, baru manifest diganti dengan generation precondition. Kegagalan training/upload mempertahankan manifest lama. Aplikasi mengecek checksum dan versi library sebelum memuat model. Jika pengambilan versi baru gagal, sesi memakai bundle terakhir yang berhasil; untuk sesi baru tersedia fallback bundle lokal lama dengan peringatan yang jelas.

Gate saat ini memeriksa validitas teknis; **belum ada ambang akurasi bisnis atau jaminan model baru lebih baik**. Metrik MAE/MSE/R2 per target dan R2 CV dicatat untuk ditinjau. Training asli tidak dapat diverifikasi sebelum credentials dipasang dan workflow pertama berjalan.

Versi model lama tetap tersimpan untuk rollback. Untuk rollback, arahkan latest.json ke object/generation/checksum dan versi library bundle lama yang cocok. Model bundle menggunakan joblib: hanya muat artifact dari bucket terpercaya ini.

## Menjalankan manual

```sh
python -m pip install -r requirements-training.txt
python train.py --project klamby-469003 --bucket klamby_sales --publish
```

Tanpa `--publish`, hasil hanya disimpan ke work/training. Kredensial menggunakan GOOGLE_APPLICATION_CREDENTIALS atau Application Default Credentials. Jalankan `python -m pytest -q` untuk pengujian data sintetis, logika kemiripan, pergantian model, dan UI Streamlit. Tes memakai grid kecil; produksi memakai seluruh grid notebook.

Untuk Colab, notebook revisi dan export_colab.py memakai best_rf_mod yang sudah dilatih. Jalankan dari awal setelah perubahan filter 90 hari; jangan hanya ekspor objek lama. Export menyertakan referensi baris X_train dan bisa diunggah dengan helper model_store.py. Otomasi bulanan tidak bergantung pada runtime Colab.

Referensi: [GitHub schedules](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule), [Google auth action](https://github.com/google-github-actions/auth), [GCS preconditions](https://docs.cloud.google.com/storage/docs/request-preconditions).
