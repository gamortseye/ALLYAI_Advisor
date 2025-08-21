#!/usr/bin/env python3
"""
Updated app.py that supports two modes:

1) ASGI FastAPI app exported as `fastapi_app` (recommended).
   - Run with uvicorn:   uvicorn app:fastapi_app --host 0.0.0.0 --port $PORT
   - Or run with gunicorn ASGI worker:
       gunicorn -k uvicorn.workers.UvicornWorker app:fastapi_app -b 0.0.0.0:$PORT

2) Minimal WSGI fallback exported as `app` (so default `gunicorn app:app` won't crash).
   - This WSGI fallback returns a friendly HTML landing page and a /health JSON route.
   - It's intentionally simple; real API endpoints use the FastAPI app.

Environment:
- MODEL_PATH: path to GGUF model (default ./model.gguf)
- PORT / HF_PORT / GRADIO_SERVER_PORT: port to bind (defaults to 8080)
- ENABLE_GRADIO=0 to disable Gradio UI in ASGI mode.
"""

import os
import json
import logging
import traceback
from typing import Optional, Any, Dict

# Logging
logger = logging.getLogger("allyai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Config
MODEL_PATH = os.environ.get("MODEL_PATH", "./model.gguf")
DEFAULT_PORT = int(os.environ.get("PORT") or os.environ.get("HF_PORT") or os.environ.get("GRADIO_SERVER_PORT") or 8080)
ENABLE_GRADIO = os.environ.get("ENABLE_GRADIO", "1") != "0"

# Try import llama-cpp-python (ASGI app will use it). If import fails, keep running and surface error via /health.
llama_available = False
llama_import_error: Optional[str] = None
Llama = None
llm_instance = None

try:
    from llama_cpp import Llama  # type: ignore
    llama_available = True
    logger.info("llama_cpp imported successfully.")
except Exception as e:
    llama_import_error = "".join(traceback.format_exception_only(type(e), e)).strip()
    logger.warning("llama_cpp import failed: %s", llama_import_error)
    logger.debug("Full import traceback:\n%s", traceback.format_exc())

# -----------------------
# ASGI (FastAPI) app
# -----------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

fastapi_app = FastAPI(title="ALLYAI (FastAPI ASGI)", version="0.1.0")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    user_input: str
    system_message: Optional[str] = None
    max_tokens: Optional[int] = 250
    temperature: Optional[float] = 0.7


def _make_full_prompt(system_message: Optional[str], user_input: str) -> str:
    sys_msg = (system_message or "").strip()
    if sys_msg:
        return f"{sys_msg}\n\nUser: {user_input}\nLawyer:"
    return f"User: {user_input}\nLawyer:"


def load_llama_model(model_path: str):
    """
    Attempt to instantiate Llama; returns instance or None.
    """
    global Llama, llm_instance
    if not llama_available or Llama is None:
        logger.error("llama-cpp python not importable: %s", llama_import_error)
        return None

    if llm_instance is not None:
        return llm_instance

    if not os.path.exists(model_path):
        logger.error("Model file not found at: %s", model_path)
        return None

    try:
        logger.info("Loading Llama model from: %s", model_path)
        llm_instance = Llama(model_path=model_path)
        logger.info("Llama model loaded.")
        return llm_instance
    except Exception as e:
        logger.exception("Failed to load Llama model: %s", e)
        return None


def generate_with_llama(model, prompt: str, max_tokens: int = 250, temperature: float = 0.7) -> str:
    """
    Call llama-cpp's create and return text.
    Defensive of return shapes across llama-cpp versions.
    """
    try:
        out = model.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        if isinstance(out, dict):
            choices = out.get("choices") or []
            if len(choices) > 0:
                choice = choices[0]
                text = choice.get("text") or choice.get("message", {}).get("content") or ""
                return text.strip()
        return str(out)
    except Exception:
        logger.exception("Generation error")
        raise


@fastapi_app.get("/health")
async def asgi_health():
    return {
        "ok": True,
        "llama_cpp_available": llama_available,
        "llama_import_error": llama_import_error,
        "MODEL_PATH": MODEL_PATH,
        "model_loaded": llm_instance is not None,
    }


@fastapi_app.post("/predict")
async def predict(req: PredictRequest):
    prompt = _make_full_prompt(req.system_message, req.user_input)
    model = load_llama_model(MODEL_PATH)
    if model is None:
        msg = (
            "Model not available. Check /health. "
            "Common reason: installed llama-cpp-python wheel is incompatible with the platform (musl vs glibc)."
        )
        logger.error(msg)
        raise HTTPException(status_code=503, detail=msg)

    try:
        text = generate_with_llama(model, prompt, max_tokens=req.max_tokens, temperature=req.temperature)
        return {"response": text, "meta": {"model_path": MODEL_PATH}}
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


# Optional Gradio UI (only used when running ASGI with uvicorn locally)
GRADIO_AVAILABLE = False
demo = None
try:
    if ENABLE_GRADIO:
        import gradio as gr  # type: ignore
        GRADIO_AVAILABLE = True

        def gr_generate(user_input: str, system_message: str, max_tokens: int, temperature: float) -> str:
            # synchronous call into FastAPI predict function
            req = PredictRequest(user_input=user_input, system_message=system_message, max_tokens=max_tokens, temperature=temperature)
            import asyncio
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(predict(req))
            return result["response"]

        with gr.Blocks() as demo:
            gr.Markdown("# ALlyAI Test UI")
            with gr.Row():
                with gr.Column(scale=3):
                    user_in = gr.Textbox(label="User Input", lines=6)
                    system_in = gr.Textbox(label="System Message (optional)", lines=3)
                    max_tokens = gr.Slider(minimum=16, maximum=1024, step=1, value=250, label="max_tokens")
                    temperature = gr.Slider(minimum=0.0, maximum=1.2, step=0.01, value=0.7, label="temperature")
                    btn = gr.Button("Generate")
                with gr.Column(scale=2):
                    out = gr.Textbox(label="Model Response", lines=20)

            btn.click(fn=gr_generate, inputs=[user_in, system_in, max_tokens, temperature], outputs=[out])

        logger.info("Gradio UI available.")
    else:
        logger.info("Gradio disabled via ENABLE_GRADIO=0.")
except Exception as e:
    logger.warning("Gradio setup failed: %s", e)
    demo = None
    GRADIO_AVAILABLE = False

logger.info("ASGI configuration: MODEL_PATH=%s, PORT=%s, ENABLE_GRADIO=%s", MODEL_PATH, DEFAULT_PORT, ENABLE_GRADIO)

# -----------------------
# WSGI fallback 'app' for hosts that call gunicorn app:app
# -----------------------
def _json_response(start_response, status_code: int, payload: Dict[str, Any]):
    body = json.dumps(payload).encode("utf-8")
    start_response(f"{status_code} OK", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
    return [body]


def wsgi_app(environ, start_response):
    """
    Minimal WSGI app exported as `app` at module-level so 'gunicorn app:app' works.
    - /health -> JSON with diagnostic
    - / -> simple HTML page instructing to run ASGI mode for full functionality
    - other -> 404
    """
    path = environ.get("PATH_INFO", "/")
    try:
        if path == "/health":
            payload = {
                "ok": True,
                "mode": "wsgi_fallback",
                "instructions": "Use an ASGI server (uvicorn) or run gunicorn with the UvicornWorker to enable the full FastAPI app.",
                "recommendation": "gunicorn -k uvicorn.workers.UvicornWorker app:fastapi_app -b 0.0.0.0:$PORT",
                "llama_cpp_available": llama_available,
                "llama_import_error": llama_import_error,
                "MODEL_PATH": MODEL_PATH,
            }
            return _json_response(start_response, 200, payload)

        # root landing page
        if path == "/" or path == "":
            html = f"""<html>
<head><title>ALLYAI - WSGI fallback</title></head>
<body>
  <h2>ALLYAI - WSGI fallback</h2>
  <p>This process is running the WSGI fallback. The full API is an ASGI FastAPI app (exported as <code>fastapi_app</code>).</p>
  <p><strong>To enable the real app:</strong></p>
  <ol>
    <li>Start with Uvicorn: <code>uvicorn app:fastapi_app --host 0.0.0.0 --port {DEFAULT_PORT}</code></li>
    <li>or run Gunicorn with the Uvicorn worker: <code>gunicorn -k uvicorn.workers.UvicornWorker app:fastapi_app -b 0.0.0.0:{DEFAULT_PORT}</code></li>
  </ol>
  <p>Health endpoint (JSON): <a href="/health">/health</a></p>
</body>
</html>"""
            body = html.encode("utf-8")
            start_response("200 OK", [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))])
            return [body]

        # not found
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"Not found"]
    except Exception:
        start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
        tb = traceback.format_exc().encode("utf-8")
        return [tb]


# Make sure module-level 'app' exists (many platforms expect 'app' to be a WSGI app).
# If the host runs `gunicorn app:app` this will be used and will not crash.
app = wsgi_app

# Expose the real ASGI FastAPI under a different name so you can run it with uvicorn or use UvicornWorker
# (e.g. uvicorn app:fastapi_app OR gunicorn -k uvicorn.workers.UvicornWorker app:fastapi_app)
# fastapi_app is already defined above.

# -----------------------
# If this file is run directly, start uvicorn serving the ASGI app.
# -----------------------
if __name__ == "__main__":
    # prefer uvicorn when running locally
    import uvicorn
    port = DEFAULT_PORT
    logger.info("Starting uvicorn for ASGI app (fastapi_app) on port %s", port)

    # If Gradio was created, launch it too (demo.launch will block)
    if GRADIO_AVAILABLE and demo is not None:
        try:
            # Launch gradio UI on the same host/port (it uses an internal server)
            demo.launch(server_name="0.0.0.0", server_port=port, share=False, prevent_thread_lock=True)
        except Exception:
            logger.exception("Gradio launch failed; falling back to uvicorn.")
            uvicorn.run("app:fastapi_app", host="0.0.0.0", port=port, log_level="info")
    else:
        uvicorn.run("app:fastapi_app", host="0.0.0.0", port=port, log_level="info")
