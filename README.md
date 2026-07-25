# MenYomi (メニヨミ) 🍜

**Read any Japanese menu.** Snap a photo of a Japanese restaurant menu and MenYomi turns it into a live bilingual menu — it reads each dish with OCR, translates it to English, adds short descriptions and cultural notes, flags allergens, and even speaks the Japanese pronunciation aloud.

Built as a solo project at the **Agent Forge AI Hackathon: Tokyo**, integrating six partner platforms end to end.

---

## What it does

- 📷 **Upload a menu photo** → get a clean, styled bilingual menu in seconds
- 🔤 **OCR** reads the Japanese dishes and prices straight from the image
- 🌐 **Translation** adds natural English names and short descriptions
- 🇯🇵 **Japanese descriptions** for local diners (both directions)
- 🏮 **Cultural notes** — a one-line explanation of each dish for tourists
- ⚠️ **Allergen flags** and **dietary filters** (vegetarian, no pork, no shellfish, etc.)
- 🖼️ **Dish photos** pulled in per item
- 🔊 **Voice pronunciation** of each dish name (browser text-to-speech)
- 📄 **Download** the finished menu as a standalone HTML page

---

## Tech stack & the six integrations

| Part | Platform | Role in MenYomi |
|------|----------|-----------------|
| OCR (primary) | **Nosana** | Self-hosted Nanonets OCR 2 model on a decentralized GPU reads the menu image |
| OCR (fallback) / vision | **Qwen Cloud** | Reads the menu image if Nosana is unavailable |
| Translation & enrichment | **GMI Cloud** | Translates dishes, writes English descriptions, cultural notes, allergens |
| Japanese descriptions | **ai&** | Natural Japanese dish descriptions on Japan-based inference |
| Code sandbox | **Daytona** | Generates and validates the downloadable HTML menu inside a secure sandbox |
| Built in | **Qoder** | The agentic IDE the whole project was written in |

Backend is **FastAPI** (Python). Frontend is a single HTML page with vanilla JavaScript. Deployed live on **Render**.

All AI providers are accessed through one small OpenAI-compatible wrapper (`providers.py`), so each is a `base_url` + `model` swap.

---

## Requirements

### Software
- **Python 3.11+** (developed on 3.12 / 3.14)
- **git**
- A terminal (Linux/macOS/WSL)

### Accounts & API keys
To run MenYomi yourself you'll need accounts and keys from these providers. All had free credits at hackathon time; check each site for current pricing.

| Provider | Sign up at | What you need |
|----------|-----------|---------------|
| GMI Cloud | https://console.gmicloud.ai | An **inference** API key |
| Qwen Cloud | https://home.qwencloud.com/api-keys | An API key |
| ai& | https://www.aiand.com | API key **and** the OpenAI-compatible base URL |
| Daytona | https://app.daytona.io | An API key |
| Nosana *(optional)* | https://dashboard.nosana.com | No API key — you deploy an OCR model and copy its endpoint URL |
| Qoder *(optional)* | https://qoder.com | Only needed if you want to develop in the same IDE |

**Nosana is optional to run the app.** If you don't set it up, MenYomi automatically falls back to Qwen for OCR — everything still works. See [Optional: Nosana OCR](#optional-nosana-self-hosted-ocr) below.

---

## Setup

### 1. Clone and enter the project
```bash
git clone https://github.com/carlmasters02/MenYomi.git
cd MenYomi
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API keys
Create a file named `.env` in the project root (copy the template below and fill in your own keys):

```env
# GMI Cloud — inference key from console.gmicloud.ai
GMI_API_KEY=your_gmi_key_here
GMI_MODEL=anthropic/claude-sonnet-5

# Qwen Cloud — key from home.qwencloud.com
QWEN_API_KEY=your_qwen_key_here
QWEN_MODEL=qwen3.6-flash

# ai& — key and base URL from aiand.com
AIAND_API_KEY=your_aiand_key_here
AIAND_BASE_URL=your_aiand_base_url_here
AIAND_MODEL=qwen/qwen3.6-27b

# Daytona — key from app.daytona.io
DAYTONA_API_KEY=your_daytona_key_here

# Nosana — OPTIONAL. Leave blank to use Qwen for OCR instead.
NOSANA_URL=
NOSANA_API_KEY=EMPTY
NOSANA_MODEL=nanonets/Nanonets-OCR2-3B
```

> ⚠️ **Never commit `.env`.** It contains your secret keys. The included `.gitignore` already excludes it — keep it that way. If a key is ever exposed, regenerate it in that provider's dashboard immediately.

### 5. (Optional) Verify your keys work
```bash
python test_keys.py
```
This makes a tiny call to each provider and reports PASS / FAIL / SKIP.

### 6. Run it
```bash
uvicorn main:app --reload
```
Open **http://127.0.0.1:8000** in your browser, upload a Japanese menu photo, and click **Translate**.

---

## Optional: Nosana self-hosted OCR

MenYomi can use a self-hosted OCR model on Nosana's decentralized GPUs as its primary menu reader (with Qwen as automatic fallback).

1. Go to https://dashboard.nosana.com → **Deploy**.
2. Choose the **Nanonets OCR 2 Models** template (3B).
3. Pick a GPU with **enough VRAM** — a 24 GB card (e.g. NVIDIA 3090) works; smaller cards like a 10 GB 3080 will run out of memory on this model.
4. Create the deployment and wait for it to finish loading (watch the Logs tab for `Application startup complete`).
5. Copy the endpoint URL and put it in `.env` as `NOSANA_URL`, adding `/v1` to the end:
   ```env
   NOSANA_URL=https://<your-endpoint>.node.k8s.prd.nos.ci/v1
   ```
6. Restart the app. Uploads will now print `OCR engine used: nosana` in the terminal.

> 💡 Nosana bills while the GPU runs and auto-stops after its container timeout. **Stop the deployment when you're done** to save credits. When Nosana is down, MenYomi silently falls back to Qwen.

---

## Deploying to Render

1. Push your repo to GitHub.
2. On https://render.com → **New → Web Service** → connect the repo.
3. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance type:** Free
4. Under **Environment**, add every variable from your `.env` (Render can't read the `.env` file — it's gitignored). Enter each key/value pair in the dashboard.
5. Create the service. Render builds and gives you a public URL.

> The included `Procfile` and `runtime.txt` (pinning Python 3.12) help Render build reliably.

---

## Project structure

```
MenYomi/
├── main.py            # FastAPI app: endpoints, OCR→translate→enrich pipeline, frontend HTML
├── providers.py       # OpenAI-compatible wrapper for GMI, Qwen, ai&, Nosana
├── sandbox.py         # Daytona sandbox helper (validates generated HTML)
├── test_keys.py       # Smoke-tests every provider key
├── requirements.txt   # Python dependencies
├── Procfile           # Start command for Render
├── runtime.txt        # Pins Python version for deploy
├── .gitignore         # Excludes .env and venv
└── README.md
```

---

## How the pipeline works

1. **Upload** — user submits a menu image to `POST /parse-menu`.
2. **OCR** — Nosana (or Qwen fallback) extracts dishes, prices, and sections as JSON.
3. **Translate** — GMI adds English names, descriptions, and allergens.
4. **Localize** — ai& adds natural Japanese descriptions.
5. **Contextualize** — GMI adds a short cultural note per dish.
6. **Render** — the frontend displays styled bilingual cards with photos, filters, and voice.
7. **Export** — `POST /export-menu` has GMI generate a standalone HTML menu, validated in a Daytona sandbox, then downloaded by the user.

The OCR step tries Nosana first and falls back to Qwen on any error, so the app never breaks if the self-hosted GPU is unavailable.

---

## Notes & limitations

- Allergen detection is model-inferred and not guaranteed accurate — **do not rely on it for real dietary safety.**
- Voice pronunciation depends on your browser/OS having a Japanese TTS voice (Chrome/Chromium bundles one).
- Dish photos are keyword-matched and thematic, not exact photos of each specific dish.
- First request to a sleeping free-tier Render instance takes ~30 seconds to wake.

---

## License

Personal / educational project. Check each partner platform's own terms for usage limits and pricing.
