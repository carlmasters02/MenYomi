import os, sys
from dotenv import load_dotenv
load_dotenv()

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []

def report(name, status, detail=""):
    results.append(status)
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))

def test_llm(name, provider, pick=False):
    from providers import PROVIDERS, list_models, llm
    cfg = PROVIDERS[provider]
    if not any(os.getenv(e) for e in cfg["key_env"]):
        return report(name, SKIP, f"no key in {cfg['key_env']}")
    if provider == "aiand" and not cfg["base_url"]:
        return report(name, SKIP, "set AIAND_BASE_URL")
    try:
        models = list_models(provider)
        model = cfg["default_model"] or (models[0] if models else None)
        if pick and not cfg["default_model"]:
            print(f"      {name} models (first 10): {models[:10]}")
        out = llm(provider, "Reply with exactly: OK", model=model, max_tokens=10)
        report(name, PASS, f"model={model}, reply={out.strip()[:30]!r}")
    except Exception as e:
        report(name, FAIL, str(e)[:200])

def test_daytona():
    if not os.getenv("DAYTONA_API_KEY"):
        return report("Daytona", SKIP, "DAYTONA_API_KEY not set")
    try:
        from sandbox import run_code
        out = run_code("print(21*2)")
        report("Daytona", PASS if "42" in (out or "") else FAIL, f"output: {out!r}")
    except Exception as e:
        report("Daytona", FAIL, str(e)[:200])

if __name__ == "__main__":
    print("== key smoke test ==\n")
    test_llm("GMI Cloud", "gmi", pick=True)
    test_llm("Qwen Cloud", "qwen")
    test_llm("ai&", "aiand")
    test_daytona()
    sys.exit(1 if FAIL in results else 0)
