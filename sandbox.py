import os
from daytona import Daytona, DaytonaConfig
from dotenv import load_dotenv

load_dotenv()

_daytona = None


def _get():
    global _daytona
    if _daytona is None:
        key = os.getenv("DAYTONA_API_KEY")
        if not key:
            raise RuntimeError("DAYTONA_API_KEY not set")
        _daytona = Daytona(DaytonaConfig(api_key=key))
    return _daytona


def run_code(code):
    sb = _get().create()
    try:
        return sb.process.code_run(code).result
    finally:
        try:
            sb.delete()
        except Exception:
            pass
