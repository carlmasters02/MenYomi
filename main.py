import json
import os
import re

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from providers import llm, vision, PROVIDERS
from sandbox import run_code

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

JP_DESC_PROMPT = (
    "For each menu item in this JSON, add a field 'jp_desc' — "
    "a short natural Japanese description (max 15 characters) of the dish for local diners. "
    "Return ONLY the same JSON with jp_desc added to each item. "
    "No markdown, no commentary."
)

CULTURE_PROMPT = (
    "For each menu item, add a field 'culture_note' — "
    "one short sentence (max 20 words) explaining what the dish is or how it's traditionally eaten, "
    "written for a foreign tourist. Return ONLY the same JSON with culture_note added to each item. "
    "No markdown, no commentary."
)

EXPORT_PROMPT = (
    "Generate a complete standalone HTML document for this restaurant menu. "
    "Include inline CSS, show each item's Japanese name, English name, price, "
    "and descriptions. Return ONLY raw HTML, no markdown fences."
)


def _strip_fences(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE)


def _salvage_items(text: str) -> list | None:
    """Try to extract complete items from a truncated JSON response."""
    cleaned = _strip_fences(text)
    # Full parse attempt
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "items" in data:
            return data["items"]
    except json.JSONDecodeError:
        pass

    # Locate the items array: prefer "items": [ ... , fall back to first [
    m = re.search(r'"items"\s*:\s*\[', cleaned)
    array_start = m.end() if m else (cleaned.find("[") + 1 or None)
    if not array_start:
        return None

    body = cleaned[array_start:]

    # Walk characters tracking brace depth (string-aware) to find last complete {}
    depth = 0
    last_complete = -1
    in_string = False
    escape_next = False
    for i, ch in enumerate(body):
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete = i

    if last_complete < 0:
        return None

    # Slice up to the last complete object and wrap in an array
    fragment = body[: last_complete + 1]
    try:
        items = json.loads(f"[{fragment}]")
        return items if isinstance(items, list) else None
    except json.JSONDecodeError:
        return None


def _ocr(image_bytes: bytes) -> str:
    """Try Nosana first; fall back to Qwen if it fails, is empty, or isn't configured."""
    nosana_url = os.getenv("NOSANA_URL", "")
    if nosana_url:
        try:
            result = vision("nosana", image_bytes, PROMPT, model=os.getenv("NOSANA_MODEL"))
            if result and result.strip():
                print("OCR engine used:", "nosana")
                return result
            print("Nosana returned empty — falling back to Qwen")
        except Exception as e:
            print(f"Nosana OCR failed ({e}) — falling back to Qwen")

    # Qwen fallback
    result = vision("qwen", image_bytes, PROMPT, model="qwen3-vl-plus")
    print("OCR engine used:", "qwen")
    return result


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
    # Step 1 — OCR (Nosana primary, Qwen fallback)
    try:
        raw = _ocr(image_bytes)
    except Exception as e:
        raise HTTPException(502, f"OCR failed: {e}")
    cleaned = _strip_fences(raw)
    try:
        ocr_data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"items": [], "raw": raw}

    # Normalize: handle both {"items": [...]} and bare [...] shapes
    if isinstance(ocr_data, list):
        ocr_data = {"items": ocr_data}
    items = ocr_data.get("items", [])
    if not items:
        return ocr_data

    # Step 2 — Translation / enrichment
    try:
        raw_tr = llm("gmi", TRANSLATE_PROMPT + "\n\n" + json.dumps(items), max_tokens=4000)
    except Exception as e:
        print("GMI CALL ERROR:", e)
        # Translation unavailable — return OCR-only data
        return ocr_data

    print("GMI RAW:", repr(raw_tr))

    try:
        enriched = json.loads(_strip_fences(raw_tr))
    except Exception as e:
        print("PARSE ERROR:", e)
        # Try to salvage complete items from the truncated response
        salvaged = _salvage_items(raw_tr)
        if salvaged:
            print(f"SALVAGED {len(salvaged)} complete items")
            return {"items": salvaged}
        # Nothing recoverable — keep original items so the page still shows Japanese names
        return ocr_data

    # Step 3 — Japanese description enrichment
    try:
        raw_jd = llm("aiand", JP_DESC_PROMPT + "\n\n" + json.dumps(enriched), max_tokens=4000)
    except Exception as e:
        print("AIAND CALL ERROR:", e)
        return {"items": enriched}

    print("AIAND RAW:", repr(raw_jd))

    try:
        with_jp_desc = json.loads(_strip_fences(raw_jd))
    except Exception as e:
        print("AIAND PARSE ERROR:", e)
        return {"items": enriched}

    # Step 4 — Cultural context enrichment
    try:
        raw_cn = llm("gmi", CULTURE_PROMPT + "\n\n" + json.dumps(with_jp_desc), max_tokens=4000)
    except Exception as e:
        print("GMI CULTURE CALL ERROR:", e)
        return {"items": with_jp_desc}

    print("GMI CULTURE RAW:", repr(raw_cn))

    try:
        with_culture = json.loads(_strip_fences(raw_cn))
    except Exception as e:
        print("GMI CULTURE PARSE ERROR:", e)
        return {"items": with_jp_desc}

    return {"items": with_culture}


class ExportRequest(BaseModel):
    items: list


@app.post("/export-menu")
def export_menu(req: ExportRequest):
    # Step 1 — Generate HTML via GMI
    try:
        raw_html = llm("gmi", EXPORT_PROMPT + "\n\n" + json.dumps(req.items), max_tokens=4000)
    except Exception as e:
        raise HTTPException(502, f"HTML generation failed: {e}")

    html = _strip_fences(raw_html)

    # Step 2 — Validate in Daytona sandbox
    validated_items = None
    try:
        # Build validation code that counts item names in the HTML
        names = [item.get("jp_name", "") for item in req.items if item.get("jp_name")]
        validation_code = (
            "html = " + repr(html) + "\n"
            "names = " + repr(names) + "\n"
            "count = sum(1 for n in names if n in html)\n"
            "print(count)"
        )
        result = run_code(validation_code)
        validated_items = int(result.strip())
    except Exception as e:
        print("SANDBOX VALIDATION ERROR:", e)

    return {"html": html, "validated_items": validated_items}


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>MenYomi — Read any Japanese menu</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+JP:wght@400;500;700&display=swap" rel="stylesheet" />
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #fbfbfd;
    --surface:   #ffffff;
    --surface-2: #f2f3f7;
    --text:      #16182b;
    --muted:     #6b7280;
    --muted-lt:  #9ca3af;
    --accent:    #e5484d;
    --accent-h:  #dc3d42;
    --indigo:    #5b6ee1;
    --indigo-lt: #eef0fb;
    --border:    #e8e9ee;
    --shadow-sm: 0 1px 2px rgba(22,24,43,0.04);
    --shadow:    0 2px 8px rgba(22,24,43,0.06), 0 1px 2px rgba(22,24,43,0.04);
    --shadow-lg: 0 8px 24px rgba(22,24,43,0.08), 0 2px 6px rgba(22,24,43,0.04);
    --radius:    14px;
    --radius-sm: 10px;
    --radius-xs: 6px;
  }

  body {
    font-family: 'Plus Jakarta Sans', 'Noto Sans JP', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }

  /* ── Sticky Header ──────────────────── */
  .sticky-header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(251,251,253,0.88);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 0.85rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
  }
  .header { display: flex; align-items: baseline; gap: 0.6rem; }
  .header .brand {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.03em;
  }
  .header .brand .jp {
    font-family: 'Noto Sans JP', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--accent);
    margin-left: 0.15rem;
  }
  .header .tagline {
    font-size: 0.8rem;
    color: var(--muted-lt);
    font-weight: 500;
  }

  /* ── Container ──────────────────────── */
  .container {
    max-width: 720px;
    margin: 0 auto;
    padding: 1.5rem 1.5rem 3rem;
  }

  /* ── Controls Bar ───────────────────── */
  .controls-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
  }
  .controls {
    display: flex;
    gap: 0.6rem;
    align-items: center;
    flex-wrap: wrap;
  }
  input[type="file"] {
    flex: 1;
    font-family: inherit;
    font-size: 0.85rem;
    color: var(--muted);
    min-width: 0;
  }
  input[type="file"]::file-selector-button {
    font-family: inherit;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.45rem 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    color: var(--text);
    cursor: pointer;
    margin-right: 0.6rem;
    transition: all 0.15s;
  }
  input[type="file"]::file-selector-button:hover { background: #eaebf0; }

  button {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    border: none;
    border-radius: var(--radius-sm);
    padding: 0.55rem 1.3rem;
    cursor: pointer;
    transition: all 0.15s;
    letter-spacing: -0.01em;
  }
  button:active { transform: scale(0.97); }
  button:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }
  button:focus-visible { outline: 2px solid var(--indigo); outline-offset: 2px; }

  #translateBtn { background: var(--accent); color: #fff; }
  #translateBtn:hover:not(:disabled) { background: var(--accent-h); box-shadow: 0 4px 12px rgba(229,72,77,0.3); }

  .export-row { margin-top: 0.7rem; }
  #exportBtn {
    background: var(--indigo);
    color: #fff;
    font-size: 0.82rem;
    padding: 0.5rem 1.1rem;
  }
  #exportBtn:hover:not(:disabled) { background: #4f60c9; box-shadow: 0 4px 12px rgba(91,110,225,0.3); }

  /* ── Spinner ────────────────────────── */
  .spinner-wrap { display: none; justify-content: center; align-items: center; padding: 2.5rem 0; gap: 0.8rem; flex-direction: column; }
  .spinner-wrap.active { display: flex; }
  .spinner {
    width: 32px; height: 32px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.65s linear infinite;
  }
  .spinner-label { font-size: 0.82rem; color: var(--muted-lt); font-weight: 500; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Status ─────────────────────────── */
  #status { font-size: 0.82rem; color: var(--muted); min-height: 1.2em; margin-bottom: 0.4rem; padding: 0 0.2rem; }
  .error { color: var(--accent) !important; font-weight: 600; }

  /* ── Results ────────────────────────── */
  #results { margin-top: 0.5rem; }

  .section-header {
    font-family: 'Noto Sans JP', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--indigo);
    background: var(--indigo-lt);
    padding: 0.3rem 0.85rem;
    border-radius: 100px;
    display: inline-block;
    margin: 1.6rem 0 0.65rem;
    letter-spacing: 0.02em;
  }
  .section-header:first-child { margin-top: 0.5rem; }

  /* ── Menu Card ──────────────────────── */
  .menu-item {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    overflow: hidden;
    padding: 1rem 1.3rem;
    margin-bottom: 0.7rem;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s, transform 0.2s;
    animation: fadeInUp 0.4s ease both;
  }
  .menu-item:hover { box-shadow: var(--shadow-lg); transform: translateY(-1px); }
  .menu-item:focus-within { box-shadow: 0 0 0 2px var(--indigo), var(--shadow); }

  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .menu-item:nth-child(2)  { animation-delay: 0.04s; }
  .menu-item:nth-child(3)  { animation-delay: 0.08s; }
  .menu-item:nth-child(4)  { animation-delay: 0.12s; }
  .menu-item:nth-child(5)  { animation-delay: 0.16s; }
  .menu-item:nth-child(6)  { animation-delay: 0.20s; }
  .menu-item:nth-child(7)  { animation-delay: 0.24s; }
  .menu-item:nth-child(8)  { animation-delay: 0.28s; }
  .menu-item:nth-child(9)  { animation-delay: 0.32s; }
  .menu-item:nth-child(10) { animation-delay: 0.36s; }

  /* ── Dish Photo ─────────────────────── */
  .dish-photo {
    width: calc(100% + 2.6rem);
    margin: -1rem -1.3rem 0.85rem -1.3rem;
    height: 170px;
    object-fit: cover;
    display: block;
    background: var(--surface-2);
    border-bottom: 1px solid var(--border);
  }

  /* ── Item Layout ────────────────────── */
  .item-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
  .item-names { flex: 1; }
  .jp-name-row { display: flex; align-items: center; gap: 0.45rem; }
  .jp-name {
    font-family: 'Noto Sans JP', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.35;
    letter-spacing: -0.01em;
  }
  .speak-btn {
    background: var(--surface-2);
    border: 1px solid var(--border);
    font-size: 0.85rem;
    cursor: pointer;
    padding: 0.2rem;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
    flex-shrink: 0;
    line-height: 1;
  }
  .speak-btn:hover { background: var(--indigo-lt); border-color: var(--indigo); }
  .speak-btn:active { transform: scale(0.9); }
  .speak-btn.speaking {
    background: var(--indigo);
    border-color: var(--indigo);
    animation: pulse-speak 0.8s ease-in-out infinite;
  }
  @keyframes pulse-speak {
    0%, 100% { box-shadow: 0 0 0 0 rgba(91,110,225,0.35); }
    50% { box-shadow: 0 0 0 6px rgba(91,110,225,0); }
  }
  .jp-desc {
    font-family: 'Noto Sans JP', sans-serif;
    font-size: 0.78rem;
    color: var(--muted-lt);
    margin-top: 0.15rem;
    font-weight: 400;
  }
  .en-name {
    font-size: 0.92rem;
    font-weight: 600;
    color: #3d3f52;
    margin-top: 0.2rem;
  }
  .price {
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--accent);
    white-space: nowrap;
    flex-shrink: 0;
    padding-top: 0.15rem;
    letter-spacing: -0.02em;
  }
  .en-desc {
    font-size: 0.84rem;
    color: var(--muted);
    margin-top: 0.4rem;
    line-height: 1.5;
  }

  /* ── Allergen Pills ─────────────────── */
  .allergens { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.5rem; }
  .pill {
    font-size: 0.7rem;
    font-weight: 600;
    background: var(--surface-2);
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 0.18rem 0.6rem;
    text-transform: capitalize;
    letter-spacing: 0.01em;
  }

  /* ── Culture Note ───────────────────── */
  .culture-note {
    font-size: 0.78rem;
    font-style: italic;
    color: var(--muted-lt);
    margin-top: 0.45rem;
    line-height: 1.4;
    padding-left: 0.6rem;
    border-left: 2px solid var(--border);
  }

  /* ── Dietary Filters ────────────────── */
  .filters {
    display: none;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.6rem;
    align-items: center;
  }
  .filters.visible { display: flex; }
  .filters-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--muted-lt);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    width: 100%;
    margin-bottom: 0.15rem;
  }
  .filter-btn {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.32rem 0.75rem;
    border-radius: 100px;
    border: 1.5px solid var(--border);
    background: var(--surface);
    color: var(--muted);
    cursor: pointer;
    transition: all 0.15s;
    user-select: none;
  }
  .filter-btn:hover { border-color: var(--indigo); color: var(--indigo); }
  .filter-btn:focus-visible { outline: 2px solid var(--indigo); outline-offset: 1px; }
  .filter-btn.on {
    background: var(--indigo);
    color: #fff;
    border-color: var(--indigo);
    box-shadow: 0 2px 8px rgba(91,110,225,0.25);
  }
  .filter-meta {
    display: none;
    align-items: center;
    gap: 0.85rem;
    margin-bottom: 0.7rem;
    font-size: 0.82rem;
    color: var(--muted);
    padding: 0 0.1rem;
  }
  .filter-meta.visible { display: flex; }
  .filter-meta label {
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-weight: 500;
  }
  .filter-meta input[type="checkbox"] { accent-color: var(--indigo); width: 15px; height: 15px; }
  .filter-count { font-weight: 700; color: var(--text); }

  /* ── Filter States ──────────────────── */
  .menu-item.dimmed { opacity: 0.25; pointer-events: none; filter: grayscale(0.5); }
  .menu-item.hidden { display: none; }
  .warn-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--accent);
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 100px;
    padding: 0.15rem 0.55rem;
    margin-top: 0.35rem;
    margin-right: 0.3rem;
  }

  /* ── Raw Block ──────────────────────── */
  .raw-block {
    white-space: pre-wrap;
    background: var(--surface-2);
    border: 1px solid var(--border);
    padding: 1rem 1.2rem;
    border-radius: var(--radius-sm);
    font-size: 0.82rem;
    color: var(--muted);
    font-family: 'SF Mono', 'Fira Code', monospace;
  }

  /* ── Reduced Motion ─────────────────── */
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; }
    .menu-item:hover { transform: none; }
  }

  /* ── Mobile ─────────────────────────── */
  @media (max-width: 520px) {
    .sticky-header { padding: 0.7rem 1rem; }
    .header .brand { font-size: 1.15rem; }
    .header .tagline { display: none; }
    .container { padding: 1rem 1rem 2.5rem; }
    .controls-bar { padding: 0.8rem 1rem; }
    .item-top { flex-direction: column; gap: 0.3rem; }
    .dish-photo { height: 140px; }
    .menu-item { padding: 0.85rem 1rem; }
  }
</style>
</head>
<body>
<div class="sticky-header">
  <div class="header">
    <div class="brand">MenYomi <span class="jp">メニヨミ</span></div>
    <div class="tagline">Read any Japanese menu</div>
  </div>
</div>
<div class="container">
  <div class="controls-bar">
    <div class="controls">
      <input type="file" id="fileInput" accept="image/*" />
      <button id="translateBtn" onclick="handleTranslate()">Translate</button>
    </div>
    <div class="export-row">
      <button id="exportBtn" onclick="handleExport()" disabled>Download Menu Page</button>
    </div>
  </div>
  <div id="status"></div>
  <div class="spinner-wrap" id="spinnerWrap"><div class="spinner"></div><div class="spinner-label">Reading menu…</div></div>
  <div class="filters" id="filterBar">
    <div class="filters-label">Dietary filters</div>
    <button class="filter-btn" data-filter="vegetarian" onclick="toggleFilter(this)">Vegetarian</button>
    <button class="filter-btn" data-filter="no_pork" onclick="toggleFilter(this)">No Pork</button>
    <button class="filter-btn" data-filter="no_shellfish" onclick="toggleFilter(this)">No Shellfish</button>
    <button class="filter-btn" data-filter="no_egg" onclick="toggleFilter(this)">No Egg</button>
    <button class="filter-btn" data-filter="no_dairy" onclick="toggleFilter(this)">No Dairy</button>
    <button class="filter-btn" data-filter="no_gluten" onclick="toggleFilter(this)">No Wheat/Gluten</button>
    <button class="filter-btn" data-filter="no_soy" onclick="toggleFilter(this)">No Soy</button>
  </div>
  <div class="filter-meta" id="filterMeta">
    <span class="filter-count" id="filterCount"></span>
    <label><input type="checkbox" id="hideToggle" onchange="applyFilters()" /> Hide instead of dim</label>
  </div>
  <div id="results"></div>
</div>
<script>
let _menuItems = [];
let exportPromise = null;

const FILTER_RULES = {
  vegetarian:    { allergens: ['meat','pork','beef','chicken','fish','shrimp','shellfish','egg','dashi','bonito','katsuobushi'],
                   text: ['pork','beef','chicken','bacon','ham','shrimp','prawn','crab','lobster','fish','tuna','salmon','bonito','meat','katsu','tonkotsu','chashu','yakitori'] },
  no_pork:       { allergens: ['pork','meat','bacon','ham'],
                   text: ['pork','bacon','ham','tonkotsu','chashu','char siu'] },
  no_shellfish:  { allergens: ['shellfish','shrimp','prawn','crab','lobster'],
                   text: ['shrimp','prawn','crab','lobster','shellfish','crawfish','ebi','kani'] },
  no_egg:        { allergens: ['egg'],
                   text: ['egg','tamago','omelet','omelette'] },
  no_dairy:      { allergens: ['dairy','milk','butter','cheese','cream'],
                   text: ['butter','cream','cheese','milk','dairy'] },
  no_gluten:     { allergens: ['gluten','wheat','soy sauce','noodle','ramen','udon','somen'],
                   text: ['wheat','gluten','soy sauce','noodle','ramen','udon','somen','breaded','flour'] },
  no_soy:        { allergens: ['soy','soy sauce','tofu','miso','edamame'],
                   text: ['tofu','miso','soy','edamame','natto'] },
};

function toggleFilter(btn) {
  btn.classList.toggle('on');
  applyFilters();
}

function getConflicts(item) {
  const allergens = (item.allergens || []).map(a => a.toLowerCase());
  const text = ((item.en_name || '') + ' ' + (item.en_desc || '')).toLowerCase();
  const active = [...document.querySelectorAll('.filter-btn.on')].map(b => b.dataset.filter);
  const hits = [];
  active.forEach(f => {
    const rule = FILTER_RULES[f];
    if (!rule) return;
    const matched = rule.allergens.some(a => allergens.some(al => al.includes(a)))
                 || rule.text.some(t => text.includes(t));
    if (matched) hits.push(f.replace(/^no_/, '').replace(/_/g, ' '));
  });
  return hits;
}

function applyFilters() {
  const cards = document.querySelectorAll('.menu-item[data-idx]');
  const hideMode = document.getElementById('hideToggle').checked;
  const activeFilters = document.querySelectorAll('.filter-btn.on').length;
  const filterBar = document.getElementById('filterMeta');
  const countEl  = document.getElementById('filterCount');

  if (!activeFilters) {
    filterBar.classList.remove('visible');
    cards.forEach(c => {
      c.classList.remove('dimmed', 'hidden');
      c.querySelectorAll('.warn-badge').forEach(b => b.remove());
    });
    return;
  }

  filterBar.classList.add('visible');
  let shown = 0;
  cards.forEach(c => {
    const idx = parseInt(c.dataset.idx);
    const item = _menuItems[idx];
    if (!item) return;
    const conflicts = getConflicts(item);
    // Remove old badges
    c.querySelectorAll('.warn-badge').forEach(b => b.remove());
    c.classList.remove('dimmed', 'hidden');
    if (conflicts.length) {
      conflicts.forEach(label => {
        const badge = document.createElement('span');
        badge.className = 'warn-badge';
        badge.textContent = `\u26A0 contains ${label}`;
        c.appendChild(badge);
      });
      if (hideMode) c.classList.add('hidden');
      else c.classList.add('dimmed');
    } else {
      shown++;
    }
  });
  countEl.textContent = `Showing ${shown} of ${_menuItems.length} dishes`;
}

function speakJapanese(text, btn) {
  if (!window.speechSynthesis) return;
  speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang = 'ja-JP';
  utt.rate = 0.9;
  btn.classList.add('speaking');
  utt.onend = () => btn.classList.remove('speaking');
  utt.onerror = () => btn.classList.remove('speaking');
  speechSynthesis.speak(utt);
}

function startExport(items) {
  const btn = document.getElementById('exportBtn');
  btn.textContent = 'Preparing download…';
  btn.disabled = true;
  exportPromise = (async () => {
    const res = await fetch('/export-menu', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items }),
    });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    return res.json();
  })();
  exportPromise.then(() => {
    btn.textContent = 'Download Menu Page';
    btn.disabled = false;
  }).catch(() => {
    btn.textContent = 'Download Menu Page';
    btn.disabled = false;
  });
}

async function handleTranslate() {
  const fileInput = document.getElementById('fileInput');
  const status    = document.getElementById('status');
  const results   = document.getElementById('results');
  const btn       = document.getElementById('translateBtn');
  const spinner   = document.getElementById('spinnerWrap');

  results.innerHTML = '';
  status.textContent = '';
  exportPromise = null;
  // Reset filters
  document.querySelectorAll('.filter-btn.on').forEach(b => b.classList.remove('on'));
  document.getElementById('filterBar').classList.remove('visible');
  document.getElementById('filterMeta').classList.remove('visible');

  if (!fileInput.files.length) {
    status.textContent = 'Please select an image first.';
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  btn.disabled = true;
  spinner.classList.add('active');
  status.textContent = 'Translating…';

  try {
    const res = await fetch('/parse-menu', { method: 'POST', body: formData });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    spinner.classList.remove('active');
    if (data.raw) {
      status.innerHTML = '<span class="error">OCR returned unparseable text — showing raw below.</span>';
      results.innerHTML = `<pre class="raw-block">${data.raw}</pre>`;
      return;
    }
    renderItems(data.items || []);
    _menuItems = data.items || [];
    status.textContent = '';
    if (_menuItems.length) {
      startExport(_menuItems);
    }
  } catch (err) {
    spinner.classList.remove('active');
    status.innerHTML = `<span class="error">${err.message}</span>`;
  } finally {
    btn.disabled = false;
  }
}

function renderItems(items) {
  const results = document.getElementById('results');
  if (!items.length) {
    results.innerHTML = '<p style="color:var(--muted)">No items found.</p>';
    document.getElementById('filterBar').classList.remove('visible');
    return;
  }
  // Show filter bar
  document.getElementById('filterBar').classList.add('visible');
  // Build a flat index map for filtering
  const idxMap = new Map();
  items.forEach((item, i) => idxMap.set(item, i));
  // Group by section
  const groups = {};
  items.forEach(item => {
    const sec = item.section || '';
    if (!groups[sec]) groups[sec] = [];
    groups[sec].push(item);
  });
  let html = '';
  Object.keys(groups).forEach(sec => {
    if (sec) html += `<div class="section-header">${sec}</div>`;
    groups[sec].forEach(item => {
      const idx = idxMap.get(item);
      const pills = (item.allergens && item.allergens.length)
        ? `<div class="allergens">${item.allergens.map(a => `<span class="pill">${a}</span>`).join('')}</div>`
        : '';
      html += `
        <div class="menu-item" data-idx="${idx}">
          ${item.en_name ? `<img class="dish-photo" src="https://source.unsplash.com/300x200/?${encodeURIComponent(item.en_name + ' japanese food')}" alt="${item.en_name}" loading="lazy" onerror="this.style.display='none'" />` : ''}
          <div class="item-top">
            <div class="item-names">
              <div class="jp-name-row">
                <div class="jp-name">${item.jp_name}</div>
                <button class="speak-btn" onclick="speakJapanese('${item.jp_name.replace(/'/g, "\\'")}', this)" title="Listen to pronunciation">🔊</button>
              </div>
              ${item.jp_desc ? `<div class="jp-desc">${item.jp_desc}</div>` : ''}
              ${item.en_name ? `<div class="en-name">${item.en_name}</div>` : ''}
            </div>
            ${item.price ? `<div class="price">${item.price}</div>` : ''}
          </div>
          ${item.en_desc ? `<div class="en-desc">${item.en_desc}</div>` : ''}
          ${pills}
          ${item.culture_note ? `<div class="culture-note">${item.culture_note}</div>` : ''}
        </div>`;
    });
  });
  results.innerHTML = html;
  applyFilters();
}

async function handleExport() {
  if (!exportPromise) return;
  const btn    = document.getElementById('exportBtn');
  const status = document.getElementById('status');
  btn.disabled = true;
  btn.textContent = 'Preparing download…';
  try {
    const data = await exportPromise;
    const blob = new Blob([data.html], { type: 'text/html' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'menu.html';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    status.textContent = data.validated_items != null
      ? `Downloaded — ${data.validated_items} items validated.`
      : 'Downloaded (sandbox validation unavailable).';
  } catch (err) {
    status.innerHTML = `<span class="error">${err.message}</span>`;
  } finally {
    btn.textContent = 'Download Menu Page';
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE
