import json
import os
import re

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from providers import llm, vision, PROVIDERS
from sandbox import run_code

app = FastAPI(title="Agent Forge starter")
app.mount("/static", StaticFiles(directory="static"), name="static")

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

PARSE_TEXT_PROMPT = (
    "Parse the following Japanese menu text into JSON. "
    "Return ONLY valid JSON in this exact shape: "
    '{"items":[{"jp_name":"","price":"","section":""}]}. '
    "Lines without a price are section headings — include them with an empty price string. "
    "Preserve Japanese text exactly as written. No markdown, no commentary."
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


def _enrich(items: list) -> dict:
    """Run the full enrichment pipeline: translation, jp_desc, culture notes."""
    # Step 1 — Translation
    try:
        raw_tr = llm("gmi", TRANSLATE_PROMPT + "\n\n" + json.dumps(items), max_tokens=4000)
    except Exception as e:
        print("GMI CALL ERROR:", e)
        return {"items": items}
    print("GMI RAW:", repr(raw_tr))

    try:
        enriched = json.loads(_strip_fences(raw_tr))
    except Exception as e:
        print("PARSE ERROR:", e)
        salvaged = _salvage_items(raw_tr)
        if salvaged:
            print(f"SALVAGED {len(salvaged)} complete items")
            return {"items": salvaged}
        return {"items": items}

    # Step 2 — Japanese description
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

    # Step 3 — Cultural context
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

    # Steps 2-4 — Enrichment pipeline (shared with /parse-text)
    return _enrich(items)


class ParseTextRequest(BaseModel):
    text: str


@app.post("/parse-text")
def parse_text(req: ParseTextRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(422, "No text provided")

    # Step 1 — Parse Japanese text into structured items
    try:
        raw = llm("gmi", PARSE_TEXT_PROMPT + "\n\n" + req.text, max_tokens=4000)
    except Exception as e:
        raise HTTPException(502, f"Text parsing failed: {e}")

    print("PARSE TEXT RAW:", repr(raw))

    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("PARSE TEXT JSON ERROR:", e)
        salvaged = _salvage_items(raw)
        if salvaged:
            return _enrich(salvaged)
        raise HTTPException(422, "Could not parse menu text into structured items")

    # Normalize shape
    if isinstance(data, list):
        data = {"items": data}
    items = data.get("items", [])
    if not items:
        return {"items": []}

    return _enrich(items)


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



@app.get("/")
def index():
    return FileResponse("templates/index.html")

@app.get("/stack")
def stack():
    return FileResponse("templates/stack.html")

@app.get("/about")
def about():
    return FileResponse("templates/about.html")

@app.get("/terms")
def terms():
    return FileResponse("templates/terms.html")

@app.get("/privacy")
def privacy():
    return FileResponse("templates/privacy.html")

