# 🧠 YouTube Sentiment Analyzer (Django + BERT Indonesia)

Aplikasi web berbasis Django untuk menganalisis sentimen komentar YouTube secara otomatis dalam Bahasa Indonesia. Aplikasi ini menggunakan model transformer dari Hugging Face yang dilatih khusus untuk Bahasa Indonesia dan menyediakan visualisasi hasil dalam bentuk diagram.

---

## 🎯 Fitur Utama

- 🎥 **Input Link YouTube:** Ambil komentar dari video secara otomatis.
- 🧠 **Analisis Sentimen:** Gunakan model BERT (`w11wo/indonesian-roberta-base-sentiment-classifier`) untuk klasifikasi positif, negatif, dan netral.
- 📊 **Visualisasi Data:**
  - Diagram **pie**: Persentase sentimen
  - Diagram **batang**: Kata terbanyak dari komentar
- 💾 **Penyimpanan Riwayat:** Hasil analisis tersimpan di database.
- 🔁 **Manajemen Riwayat:** Lihat, edit, atau hapus hasil analisis sebelumnya.
- 🐳 **Dockerized:** Mudah dijalankan di container.
- 🚀 **CI/CD:** Otomatis build dan test saat push dan pull request melalui GitHub Actions.

---

## 🧠 Model yang Digunakan

> [`w11wo/indonesian-roberta-base-sentiment-classifier`](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier)

- Bahasa: Indonesia 🇮🇩
- Model: RoBERTa
- Tugas: Sentiment Classification
- Kategori: Positif, Negatif, Netral

---

## 📁 Struktur Folder

```
YOUTUBE/
│
├── .github/                  # GitHub Actions CI/CD
│   └── workflows/ci-cd.yml
├── .vscode/                  # VSCode config (opsional)
├── env/                      # Virtual environment (diabaikan oleh git)
├── media/                    # File upload/media
├── Youtube/                  # Django project settings
├── youtubeApi/               # Django app untuk logika bisnis
├── .gitignore
├── docker-compose.yaml       # Docker Compose config
├── Dockerfile                # Dockerfile aplikasi Django
├── manage.py                 # Django CLI
├── requirements.txt          # Dependencies
```

---

## 🐳 Jalankan Aplikasi dengan Docker

### 🔧 Build dan Jalankan

```bash
git clone https://github.com/username/youtube-sentiment-analyzer.git
cd youtube-sentiment-analyzer
docker-compose up -d --build
```

Lalu buka di browser:
[http://localhost:8000](http://localhost:8000)

## 📦 Dependencies (`requirements.txt`)

```
django==4.2
pillow==9.5.0
torch==1.13.1
transformers==4.30.2
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
mysqlclient==2.2.0
gunicorn==20.1.0
google-api-python-client==2.171.0
```

---

## 🙌 Kontribusi

Pull request dan saran terbuka untuk siapa saja! Pastikan kamu menjalankan `docker-compose build` dan `test` sebelum melakukan PR.
