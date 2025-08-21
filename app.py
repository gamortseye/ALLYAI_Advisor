#!/usr/bin/env python3
"""
Robust app.py for serving a local gguf model with llama-cpp-python (if available),
plus a Gradio test UI and a FastAPI JSON API endpoint.

Usage:
- Configure environment variables:
    MODEL_PATH: path to your .gguf (default: "./model.gguf")
    PORT / HF_PORT / GRADIO_SERVER_PORT: port to bind (defaults to 8080)
    ENABLE_GRADIO: "0" to disable Gradio UI; otherwise Gradio is attempted if installed
- Run: python app.py
- On platforms like Render set PORT=8080; on Hugging Face Spaces set HF_PORT
"""

import os
import logging
import traceback
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("model_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Config
MODEL_PATH = os.environ.get("MODEL_PATH", "./model.gguf")
DEFAULT_PORT = int(
    os.environ.get("PORT")
    or os.environ.get("HF_PORT")
    or os.environ.get("GRADIO_SERVER_PORT")
    or 8080
)
ENABLE_GRADIO = os.environ.get("ENABLE_GRADIO", "1") != "0"

# Attempt to import llama-cpp-python
llama_available = False
llama_import_error: Optional[str] = None
Llama = None
llm_instance = None

try:
    # import may raise if wheel/platform mismatch (musl vs glibc) or missing lib
    from llama_cpp import Llama  # type: ignore
    llama_available = True
    logger.info("llama_cpp imported successfully.")
except Exception as e:
    llama_import_error = "".join(traceback.format_exception_only(type(e), e)).strip()
    logger.warning("llama_cpp import failed: %s", llama_import_error)
    logger.debug("Full import traceback:\n%s", traceback.format_exc())


# FastAPI setup
app = FastAPI(title="Local GGUF Model API (llama-cpp)", version="0.1.0")

# Allow cross origin for local testing / frontend
app.add_middleware(
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
    """
    Recreates the prompt pattern you used previously: system message + "User: ...\nLawyer:"
    """
    sys_msg = (system_message or "").strip()
    if sys_msg:
        return f"{sys_msg}\n\nUser: {user_input}\nLawyer:"
    return f"User: {user_input}\nLawyer:"


def load_llama_model(model_path: str) -> Optional["Llama"]:
    """
    Try to instantiate a llama-cpp Llama object. Return None on failure (and log).
    """
    global Llama, llm_instance
    if not llama_available or Llama is None:
        logger.error("llama-cpp python is not importable: %s", llama_import_error)
        return None

    if llm_instance is not None:
        return llm_instance

    if not os.path.exists(model_path):
        logger.error("Requested model_path does not exist: %s", model_path)
        return None

    try:
        logger.info("Loading llama-cpp model from: %s", model_path)
        # keep default params; user can edit this to add n_ctx etc.
        llm_instance = Llama(model_path=model_path)
        logger.info("Llama model loaded.")
        return llm_instance
    except Exception as e:
        logger.exception("Failed to instantiate Llama model: %s", e)
        return None


def generate_with_llama(
    model: "Llama",
    prompt: str,
    max_tokens: int = 250,
    temperature: float = 0.7,
) -> str:
    """
    Use llama-cpp Llama.create(...) and return the textual response.
    """
    # llama-cpp-python's Llama.create returns a dict with 'choices' list
    # This code is defensive about the structure.
    try:
        out = model.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        # older versions return out['choices'][0]['text']
        if isinstance(out, dict):
            choices = out.get("choices") or []
            if len(choices) > 0:
                # support both 'text' or 'message' shapes
                choice = choices[0]
                text = choice.get("text") or choice.get("message", {}).get("content") or ""
                return text.strip()
        # Fallback: stringify
        return str(out)
    except Exception:
        logger.exception("Error during llama generation")
        raise


@app.get("/health")
async def health():
    return {
        "ok": True,
        "llama_cpp_available": llama_available,
        "llama_import_error": llama_import_error,
        "model_path": MODEL_PATH,
        "model_loaded": llm_instance is not None
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    Predict endpoint: accepts JSON with user_input and optional system_message.
    Returns JSON with `response` text and debug info.
    """
    prompt = _make_full_prompt(req.system_message, req.user_input)

    # Try to ensure model is loaded
    model = load_llama_model(MODEL_PATH)
    if model is None:
        # helpful error explaining typical causes
        msg = (
            "Model not available. Either llama-cpp-python failed to import, "
            "or the model file could not be loaded. "
            "Check /health for details. "
            "Common reason: installed llama-cpp-python wheel is incompatible with the platform "
            "(musl vs glibc)."
        )
        logger.error(msg)
        raise HTTPException(status_code=503, detail=msg)

    # Generate synchronously
    try:
        text = generate_with_llama(model, prompt, max_tokens=req.max_tokens, temperature=req.temperature)
        return {
            "response": text,
            "meta": {
                "model_path": MODEL_PATH,
                "llama_cpp_available": llama_available,
            }
        }
    except Exception as e:
        logger.exception("Generation failure: %s", e)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


# --- Optional Gradio UI for quick testing ---
demo = None
GRADIO_AVAILABLE = False
try:
    if ENABLE_GRADIO:
        import gradio as gr  # type: ignore

        GRADIO_AVAILABLE = True

        def gr_generate(user_input: str, system_message: str, max_tokens: int, temperature: float) -> str:
            # reuse the same predict logic - call the local function synchronously
            req = PredictRequest(user_input=user_input, system_message=system_message, max_tokens=max_tokens, temperature=temperature)
            # In-process call of predict
            import asyncio
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(predict(req))
            return result["response"]

        with gr.Blocks() as demo:
            gr.Markdown("# Local GGUF / llama-cpp Test UI")
            with gr.Row():
                with gr.Column(scale=3):
                    user_in = gr.Textbox(label="User Input", placeholder="Describe the GBV case or question here...", lines=6)
                    system_in = gr.Textbox(label="System Message (optional)", placeholder="System: You are a legal assistant..." , lines=3)
                    max_tokens = gr.Slider(minimum=16, maximum=1024, step=1, value=250, label="max_tokens")
                    temperature = gr.Slider(minimum=0.0, maximum=1.2, step=0.01, value=0.7, label="temperature")
                    btn = gr.Button("Generate")
                with gr.Column(scale=2):
                    out = gr.Textbox(label="Model Response", lines=20)

            btn.click(fn=gr_generate, inputs=[user_in, system_in, max_tokens, temperature], outputs=[out])

        logger.info("Gradio UI is available.")
    else:
        logger.info("Gradio disabled via ENABLE_GRADIO=0")
except Exception as e:
    logger.warning("Failed to import or set up Gradio: %s", e)
    GRADIO_AVAILABLE = False
    demo = None


# helpful runtime note printed at import time
logger.info("Configuration: MODEL_PATH=%s, DEFAULT_PORT=%s, ENABLE_GRADIO=%s", MODEL_PATH, DEFAULT_PORT, ENABLE_GRADIO)


# Entrypoint
if __name__ == "__main__":
    import uvicorn
    chosen_port = DEFAULT_PORT

    logger.info("Starting server. Gradio available=%s", GRADIO_AVAILABLE)

    # If Gradio is available and enabled, launch it (good for local testing).
    # On many hosting platforms (Render/HF) the platform will look for an ASGI app,
    # so if Gradio fails we fallback to uvicorn running FastAPI.
    if GRADIO_AVAILABLE and demo is not None:
        try:
            # gradio's demo.launch blocks by default. prevent_thread_lock=True is recommended for programmatic launching.
            demo.launch(server_name="0.0.0.0", server_port=chosen_port, share=False, prevent_thread_lock=True)
        except Exception:
            logger.exception("Gradio failed to launch; falling back to uvicorn for FastAPI.")
            uvicorn.run("app:app", host="0.0.0.0", port=chosen_port, log_level="info")
    else:
        # Production path: run FastAPI via uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=chosen_port, log_level="info")
