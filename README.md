# 📘 README Proyek

## 🧠 Tujuan Proyek

AGNO Service merupakan middleware orkestrasi neural berbasis tujuan agen yang mengintegrasikan berbagai model bahasa besar (LLM) dengan kemampuan pencarian semantik untuk menciptakan agen AI yang cerdas dan terkoordinasi. Sistem ini dirancang untuk:
1. Menyediakan platform terpadu untuk mengintegrasikan berbagai penyedia LLM
2. Menerapkan pencarian semantik untuk konteks yang relevan
3. Mengelola interaksi antar agen AI secara terkoordinasi
4. Menyediakan antarmuka API yang konsisten untuk integrasi sistem
5. Mengoptimalkan penggunaan sumber daya AI dengan caching dan manajemen konteks

## 🚀 Kelebihan Pengembangan Ini

- **Integrasi Multi-Provider LLM**: Mendukung OpenAI, Groq, dan penyedia LLM lainnya
- **Pencarian Semantik Canggih**: Implementasi Milvus untuk pencarian vektor yang efisien
- **Manajemen Konteks Cerdas**: Sistem caching dan penyimpanan konteks yang terstruktur
- **Koordinasi Agen**: Mekanisme untuk mengelola interaksi antar agen AI
- **API RESTful Modern**: Endpoint yang terstruktur dan terdokumentasi dengan baik
- **Monitoring Terintegrasi**: Sistem logging dan metrik performa menggunakan Prometheus
- **Keamanan Terjamin**: Implementasi autentikasi JWT dan manajemen kredensial yang aman
- **Containerisasi**: Dukungan Docker untuk deployment yang konsisten

## 🛠 Teknologi yang Digunakan

- **FastAPI**: Framework modern untuk membangun API dengan performa tinggi
- **PyTorch & Transformers**: Untuk pemrosesan bahasa alami dan model AI
- **Milvus**: Sistem pencarian vektor untuk pencarian semantik
- **Redis**: Sistem caching dan penyimpanan data in-memory
- **OpenAI & Groq**: Integrasi dengan model bahasa canggih
- **Prometheus**: Monitoring dan metrik sistem
- **Docker**: Containerization untuk deployment yang konsisten
- **Pydantic**: Validasi data dan serialisasi
- **JWT**: Autentikasi dan otorisasi
- **Loguru**: Sistem logging yang komprehensif

## 📦 Instalasi dan Setup

1. Clone repositori:
```bash
git clone [URL_REPOSITORI]
cd agno-service
```

2. Setup lingkungan virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Konfigurasi:
```bash
cp .env.example .env
# Edit .env dengan konfigurasi yang sesuai
```

## 🚀 Menjalankan Aplikasi

### Mode Development
```bash
uvicorn src.main:app --reload
```

### Mode Production
```bash
uvicorn src.main:app --workers 4
```

### Menggunakan Docker
```bash
docker build -t agno-service .
docker-compose up -d
```

## 📚 Dokumentasi API

### Swagger UI
```
http://localhost:8000/api/v1/docs
```

### ReDoc
```
http://localhost:8000/api/v1/redoc
```

### Autentikasi
1. Klik tombol **Authorize** (ikon gembok)
2. Masukkan kredensial:
```
username: admin
password: admin123
```

## 🏗 Struktur Proyek

```
agno-service/
├── src/
│   ├── core/               # Komponen inti sistem
│   │   ├── config.py      # Manajemen konfigurasi
│   │   ├── logging.py     # Sistem logging
│   │   ├── errors.py      # Penanganan error
│   │   ├── security.py    # Keamanan dan autentikasi
│   │   ├── context.py     # Manajemen konteks
│   │   ├── providers.py   # Integrasi penyedia LLM
│   │   └── search.py      # Implementasi pencarian semantik
│   ├── agents/            # Logika dan perilaku agen
│   ├── models/            # Model data dan skema
│   ├── data/             # Dataset dan sumber data
│   ├── prompts/          # Template dan sistem prompt
│   └── main.py           # Entry point aplikasi FastAPI
├── tests/                # Unit test dan integration test
├── logs/                # File log aplikasi
├── .env.example         # Template konfigurasi
├── requirements.txt     # Dependensi Python
├── docker-compose.yml   # Konfigurasi Docker
└── Dockerfile          # Build image Docker
```

## 🧪 Testing

Jalankan test suite dengan:
```bash
pytest tests/ --cov=src
```

## 📊 Monitoring dan Pemeliharaan

1. **Logging**: File log tersedia di folder `logs/`
2. **Metrics**: Endpoint metrics di `http://localhost:8000/metrics`
3. **Health Check**: `http://localhost:8000/health`
4. **Backup**: Jalankan script backup secara berkala

## 📈 Hasil yang Diharapkan

1. **Respons Cepat**: Sistem memberikan respons dalam waktu < 1 detik untuk kebanyakan permintaan
2. **Akurasi Tinggi**: Pencarian semantik memberikan hasil yang relevan dengan konteks
3. **Skalabilitas**: Sistem dapat menangani ribuan permintaan per detik
4. **Keandalan**: Uptime > 99.9% dengan sistem failover
5. **Efisiensi**: Optimasi penggunaan token dan caching untuk mengurangi biaya

## 📝 Contoh Penggunaan

### 1. Inisialisasi dan Konfigurasi
```python
from src.core.providers import ProviderFactory
from src.core.search import SemanticSearch

# Inisialisasi penyedia LLM
provider = ProviderFactory.get_provider("openai")

# Inisialisasi pencarian semantik
search = SemanticSearch()

# Konfigurasi
provider.configure(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)
```

### 2. Pencarian Semantik
```python
# Indeks dokumen
search.index_document(
    id="doc1",
    content="Konten dokumen...",
    metadata={"source": "web", "date": "2024-01-01"}
)

# Pencarian
results = search.search(
    query="Pertanyaan pencarian",
    top_k=5
)
```

### 3. Interaksi dengan LLM
```python
# Chat completion
response = await provider.chat_completion(
    messages=[
        {"role": "system", "content": "Anda adalah asisten AI yang membantu..."},
        {"role": "user", "content": "Pertanyaan pengguna..."}
    ]
)

print(response.content)
```

## 📄 Lisensi

MIT License 