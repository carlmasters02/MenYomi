import json
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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+JP:wght@400;500;700&display=swap" rel="stylesheet" />
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #f4f1ec;
    --surface:  #ffffff;
    --text:     #1a1a1a;
    --muted:    #6b7280;
    --accent:   #d94f3d;
    --accent2:  #2a9d8f;
    --border:   #e5e1db;
    --shadow:   0 2px 12px rgba(0,0,0,0.06);
    --shadow-lg:0 8px 32px rgba(0,0,0,0.10);
    --radius:   12px;
    --radius-sm:8px;
  }

  body {
    font-family: 'Inter', 'Noto Sans JP', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    padding: 1.5rem;
    line-height: 1.5;
  }

  .container {
    background: var(--surface);
    border-radius: 16px;
    box-shadow: var(--shadow-lg);
    padding: 2rem 2rem 2.5rem;
    width: 100%;
    max-width: 640px;
    align-self: flex-start;
  }

  /* ── Header ─────────────────────────── */
  .header { text-align: center; margin-bottom: 2rem; }
  .header .brand {
    font-family: 'Noto Sans JP', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.02em;
  }
  .header .brand .jp {
    font-size: 1.4rem;
    color: var(--text);
    margin-left: 0.3rem;
  }
  .header .tagline {
    font-size: 0.95rem;
    color: var(--muted);
    margin-top: 0.25rem;
  }
  .divider {
    height: 1px;
    background: var(--border);
    margin-bottom: 1.5rem;
  }

  /* ── Controls ───────────────────────── */
  .controls {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 1rem;
    align-items: center;
    flex-wrap: wrap;
  }
  input[type="file"] {
    flex: 1;
    font-family: inherit;
    font-size: 0.9rem;
    color: var(--muted);
  }
  input[type="file"]::file-selector-button {
    font-family: inherit;
    font-size: 0.85rem;
    padding: 0.4rem 0.9rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--surface);
    color: var(--text);
    cursor: pointer;
    margin-right: 0.5rem;
    transition: background 0.15s;
  }
  input[type="file"]::file-selector-button:hover { background: #f0ece6; }

  button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    border: none;
    border-radius: var(--radius-sm);
    padding: 0.55rem 1.2rem;
    cursor: pointer;
    transition: background 0.15s, transform 0.1s;
  }
  button:active { transform: scale(0.97); }
  button:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }

  #translateBtn { background: var(--accent); color: #fff; }
  #translateBtn:hover:not(:disabled) { background: #c0412f; }

  #exportBtn {
    background: var(--accent2);
    color: #fff;
    margin-bottom: 1rem;
  }
  #exportBtn:hover:not(:disabled) { background: #228176; }

  /* ── Spinner ────────────────────────── */
  .spinner-wrap { display: none; justify-content: center; padding: 2rem 0; }
  .spinner-wrap.active { display: flex; }
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Status ─────────────────────────── */
  #status { font-size: 0.85rem; color: var(--muted); min-height: 1.2em; margin-bottom: 0.5rem; }
  .error { color: var(--accent) !important; }

  /* ── Results ────────────────────────── */
  #results { margin-top: 0.5rem; }

  .section-header {
    font-family: 'Noto Sans JP', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent);
    margin: 1.5rem 0 0.6rem;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid var(--accent);
    display: inline-block;
  }
  .section-header:first-child { margin-top: 0.5rem; }

  .menu-item {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    box-shadow: var(--shadow);
    transition: box-shadow 0.15s;
  }
  .menu-item:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.10); }

  .dish-photo {
    width: calc(100% + 2.4rem);
    margin: -1rem -1.2rem 0.75rem -1.2rem;
    height: 160px;
    object-fit: cover;
    display: block;
    background: #f0ece6;
  }

  .item-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
  .item-names { flex: 1; }
  .jp-name {
    font-family: 'Noto Sans JP', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.3;
  }
  .jp-desc {
    font-family: 'Noto Sans JP', sans-serif;
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.1rem;
  }
  .en-name {
    font-size: 0.95rem;
    font-weight: 500;
    color: #374151;
    margin-top: 0.15rem;
  }
  .price {
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent);
    white-space: nowrap;
    flex-shrink: 0;
    padding-top: 0.1rem;
  }
  .en-desc {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 0.35rem;
    line-height: 1.4;
  }
  .allergens { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.45rem; }
  .pill {
    font-size: 0.72rem;
    font-weight: 600;
    background: #fef2f0;
    color: var(--accent);
    border: 1px solid #fcd9cf;
    border-radius: 100px;
    padding: 0.15rem 0.55rem;
    text-transform: capitalize;
  }
  .culture-note {
    font-size: 0.8rem;
    font-style: italic;
    color: var(--muted);
    margin-top: 0.4rem;
    line-height: 1.35;
  }

  /* ── Dietary Filters ───────────────────── */
  .filters {
    display: none;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-bottom: 0.5rem;
    align-items: center;
  }
  .filters.visible { display: flex; }
  .filters-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--muted);
    width: 100%;
    margin-bottom: 0.15rem;
  }
  .filter-btn {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.3rem 0.65rem;
    border-radius: 100px;
    border: 1.5px solid var(--border);
    background: var(--surface);
    color: var(--muted);
    cursor: pointer;
    transition: all 0.15s;
    user-select: none;
  }
  .filter-btn:hover { border-color: #bbb; }
  .filter-btn.on {
    background: var(--accent2);
    color: #fff;
    border-color: var(--accent2);
  }
  .filter-meta {
    display: none;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: var(--muted);
  }
  .filter-meta.visible { display: flex; }
  .filter-meta label { cursor: pointer; display: flex; align-items: center; gap: 0.3rem; }
  .filter-meta input[type="checkbox"] { accent-color: var(--accent2); }
  .filter-count { font-weight: 600; }
  .menu-item.dimmed { opacity: 0.3; pointer-events: none; }
  .menu-item.hidden { display: none; }
  .warn-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    color: #fff;
    background: var(--accent);
    border-radius: 100px;
    padding: 0.1rem 0.5rem;
    margin-top: 0.35rem;
    margin-right: 0.25rem;
  }

  .raw-block {
    white-space: pre-wrap;
    background: #f9f6f1;
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    color: var(--muted);
  }

  @media (max-width: 480px) {
    body { padding: 0.75rem; }
    .container { padding: 1.25rem 1.25rem 1.75rem; }
    .header .brand { font-size: 1.6rem; }
    .item-top { flex-direction: column; gap: 0.3rem; }
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="brand">MenYomi <span class="jp">メニヨミ</span></div>
    <div class="tagline">Read any Japanese menu</div>
  </div>
  <div class="divider"></div>
  <div class="controls">
    <input type="file" id="fileInput" accept="image/*" />
    <button id="translateBtn" onclick="handleTranslate()">Translate</button>
  </div>
  <div>
    <button id="exportBtn" onclick="handleExport()" disabled>Download Menu Page</button>
  </div>
  <div id="status"></div>
  <div class="spinner-wrap" id="spinnerWrap"><div class="spinner"></div></div>
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
              <div class="jp-name">${item.jp_name}</div>
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
