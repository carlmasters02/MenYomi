import json
import re

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from providers import llm, vision, PROVIDERS

app = FastAPI(title="Agent Forge starter")

PROMPT = (
    "You are an OCR engine for Japanese restaurant menus. Extract every dish. "
    "Return ONLY valid JSON, no markdown fences, in this exact shape: "
    '{"items":[{"jp_name":"","price":"","section":""}]}. '
    "Preserve Japanese text exactly. If a price is unreadable use an empty string."
)

TRANSLATE_PROMPT = (
    "You translate Japanese menu items for foreign diners. "
    "The input is JSON with a list of items, each having jp_name, price, section. "
    "For each item add three fields: en_name (natural English name), "
    "en_desc (max 12 words describing the dish), and allergens (array of strings, "
    "empty array if none). Return ONLY the same JSON structure with these fields "
    "added to each item. No commentary, no markdown fences."
)


def _strip_fences(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE)

class AgentRequest(BaseModel):
    prompt: str
    provider: str = "qwen"
    model: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/agent")
def agent(req: AgentRequest):
    if req.provider not in PROVIDERS:
        raise HTTPException(400, f"provider must be one of {list(PROVIDERS)}")
    try:
        return {"reply": llm(req.provider, req.prompt, model=req.model)}
    except Exception as e:
        raise HTTPException(502, str(e))

@app.post("/parse-menu")
async def parse_menu(file: UploadFile = File(...)):
    image_bytes = await file.read()
    # Step 1 — OCR
    try:
        raw = vision("qwen", image_bytes, PROMPT, model="qwen3-vl-plus")
    except Exception as e:
        raise HTTPException(502, f"OCR failed: {e}")
    cleaned = _strip_fences(raw)
    try:
        ocr_data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"items": [], "raw": raw}

    items = ocr_data.get("items", [])
    if not items:
        return ocr_data

    # Step 2 — Translation / enrichment
    try:
        raw_tr = llm("gmi", TRANSLATE_PROMPT + "\n\n" + json.dumps(items))
    except Exception as e:
        print("GMI CALL ERROR:", e)
        # Translation unavailable — return OCR-only data
        return ocr_data

    print("GMI RAW:", repr(raw_tr))

    try:
        enriched = json.loads(_strip_fences(raw_tr))
    except Exception as e:
        print("PARSE ERROR:", e)
        # Parse failed — keep original items so the page still shows Japanese names
        return ocr_data

    return {"items": enriched}

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Menu Translator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f5f5f5; min-height: 100vh; display: flex; justify-content: center; padding: 2rem; }
  .container { background: #fff; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.1); padding: 2rem; width: 100%; max-width: 600px; }
  h1 { font-size: 1.5rem; margin-bottom: 1.5rem; color: #333; }
  .controls { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; align-items: center; flex-wrap: wrap; }
  input[type="file"] { flex: 1; }
  button { background: #e63946; color: #fff; border: none; border-radius: 8px; padding: 0.6rem 1.4rem; font-size: 1rem; cursor: pointer; transition: background 0.2s; }
  button:hover { background: #c1121f; }
  button:disabled { background: #aaa; cursor: not-allowed; }
  #status { font-size: 0.9rem; color: #555; min-height: 1.2em; }
  #results { margin-top: 1rem; }
  .menu-item { border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem; background: #fafafa; }
  .menu-item .jp { font-size: 1.2rem; font-weight: bold; color: #333; }
  .menu-item .en { font-size: 1rem; color: #555; margin-top: 0.2rem; }
  .menu-item .price { font-size: 0.95rem; color: #e63946; font-weight: 600; margin-top: 0.3rem; }
  .menu-item .desc { font-size: 0.88rem; color: #777; margin-top: 0.3rem; }
  .error { color: #c1121f; font-size: 0.9rem; }
</style>
</head>
<body>
<div class="container">
  <h1>🍜 Menu Translator</h1>
  <div class="controls">
    <input type="file" id="fileInput" accept="image/*" />
    <button id="translateBtn" onclick="handleTranslate()">Translate</button>
  </div>
  <div id="status"></div>
  <div id="results"></div>
</div>
<script>
async function handleTranslate() {
  const fileInput = document.getElementById('fileInput');
  const status   = document.getElementById('status');
  const results  = document.getElementById('results');
  const btn      = document.getElementById('translateBtn');

  results.innerHTML = '';
  status.textContent = '';

  if (!fileInput.files.length) {
    status.textContent = 'Please select an image first.';
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  btn.disabled = true;
  status.textContent = 'Translating…';

  try {
    const res = await fetch('/parse-menu', { method: 'POST', body: formData });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    if (data.raw) {
      status.innerHTML = '<span class="error">OCR returned unparseable text — showing raw below.</span>';
      results.innerHTML = `<pre style="white-space:pre-wrap;background:#f0f0f0;padding:1rem;border-radius:8px;font-size:0.85rem;">${data.raw}</pre>`;
      return;
    }
    renderItems(data.items || []);
    status.textContent = '';
  } catch (err) {
    status.innerHTML = `<span class="error">${err.message}</span>`;
  } finally {
    btn.disabled = false;
  }
}

function renderItems(items) {
  const results = document.getElementById('results');
  if (!items.length) {
    results.innerHTML = '<p>No items found.</p>';
    return;
  }
  results.innerHTML = items.map(item => `
    <div class="menu-item">
      <div class="jp">${item.jp_name}</div>
      ${item.en_name ? `<div class="en">${item.en_name}</div>` : ''}
      ${item.section ? `<div style="font-style:italic;color:#888;font-size:0.85rem;">${item.section}</div>` : ''}
      <div class="price">${item.price ? '¥' + item.price : ''}</div>
      ${item.en_desc ? `<div class="desc">${item.en_desc}</div>` : ''}
      ${item.allergens && item.allergens.length ? `<div style="font-size:0.8rem;color:#c1121f;margin-top:0.2rem;">Allergens: ${item.allergens.join(', ')}</div>` : ''}
    </div>
  `).join('');
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE
