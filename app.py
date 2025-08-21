#!/usr/bin/env python3
"""
app.py

Robust startup for hosting a local gguf (llama.cpp) model with:
 - automatic writable model path selection
 - auto-pick best local .gguf (q4_K_M preferred)
 - try to load llama-cpp-python Llama backend (if available)
 - FastAPI endpoint /api/generate for frontend
 - Gradio UI for local testing

Environment variables:
 - MODEL_LOCAL_PATH  (optional) directory to store model files
 - HF_TOKEN          (optional) HF token for private repos/downloads
 - HF_REPO_ID        (optional) "username/repo" to download model from automatically
 - MODEL_FILENAME    (optional) exact filename in repo (overrides selection)
 - GRADIO_SERVER_PORT (optional) preferred start port (default 7861)
 - HF_PORT           (optional) alias used by Spaces (we respect either)
"""

import os
import sys
import tempfile
import logging
import socket
import glob
import json
from typing import Optional, Dict, Any

# Logging --------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("app")

# Config / Model path helpers ------------------------------------------------
def choose_writable_model_dir() -> str:
    # Prefer env var if set
    env_path = os.environ.get("MODEL_LOCAL_PATH")
    candidates = []
    if env_path:
        candidates.append(env_path)
    # prefer current working dir (project) then /tmp
    candidates.append(os.path.join(os.getcwd(), "models"))
    candidates.append(os.path.join(tempfile.gettempdir(), "models"))
    for p in candidates:
        try:
            os.makedirs(p, exist_ok=True)
            # write test
            testfile = os.path.join(p, ".writetest")
            with open(testfile, "w") as f:
                f.write("ok")
            os.remove(testfile)
            logger.info("Using model path: %s", p)
            return p
        except Exception as e:
            logger.warning("Cannot use path %s: %s", p, e)
    logger.critical("No writable path available for model files. Exiting.")
    sys.exit(1)

MODEL_LOCAL_PATH = choose_writable_model_dir()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID")  # e.g. "Gamortsey/AllyAI-lawchat-gguf"
MODEL_FILENAME = os.environ.get("MODEL_FILENAME")  # exact filename if you want
# Port selection (try HF_PORT or GRADIO_SERVER_PORT or default 7861)
DEFAULT_PORT = int(os.environ.get("HF_PORT") or os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("PORT") or 7861)

# Model selection ------------------------------------------------------------
def select_local_gguf_model(local_dir: str, prefer: Optional[list] = None) -> Optional[str]:
    """
    Return the full path to the preferred gguf file in local_dir, or None.
    Preference order default: q4_K_M, q4_0, q4_K_S, f16, *.gguf
    """
    prefer = prefer or ["q4_K_M", "q4_0", "q4_K_S", "f16"]
    ggufs = sorted(glob.glob(os.path.join(local_dir, "*.gguf")), key=os.path.getsize)
    if not ggufs:
        return None
    # try preferences first (substring match)
    for p in prefer:
        for f in ggufs:
            if p in os.path.basename(f):
                return f
    # fallback: smallest file (likely quantized)
    return ggufs[0]

LOCAL_MODEL_PATH = None
if MODEL_FILENAME:
    candidate = os.path.join(MODEL_LOCAL_PATH, MODEL_FILENAME)
    if os.path.exists(candidate):
        LOCAL_MODEL_PATH = candidate
    else:
        logger.warning("MODEL_FILENAME provided but not present locally: %s", candidate)
if LOCAL_MODEL_PATH is None:
    LOCAL_MODEL_PATH = select_local_gguf_model(MODEL_LOCAL_PATH)
if LOCAL_MODEL_PATH:
    logger.info("Selected local model: %s", LOCAL_MODEL_PATH)
else:
    logger.info("No local .gguf files found in %s", MODEL_LOCAL_PATH)

# Try to import llama-cpp-python ------------------------------------------------
llama_backend_available = False
llm = None
try:
    from llama_cpp import Llama  # type: ignore
    llama_backend_available = True
    logger.info("llama-cpp-python import OK (Llama backend available).")
except Exception as e:
    logger.warning("llama-cpp-python not available or failed to import: %s", e)
    llama_backend_available = False

# Model loader / wrapper ------------------------------------------------------
def load_llama_model(path: str):
    global llm, llama_backend_available
    if not llama_backend_available:
        raise RuntimeError("llama-cpp-python backend not available in this environment.")
    logger.info("Loading model via llama-cpp-python: %s", path)
    try:
        llm = Llama(model_path=path)
        logger.info("Model loaded successfully.")
        return True
    except Exception as e:
        logger.error("Failed to instantiate Llama model: %s", e, exc_info=True)
        llm = None
        return False

if LOCAL_MODEL_PATH and llama_backend_available and llm is None:
    ok = load_llama_model(LOCAL_MODEL_PATH)
    if not ok:
        logger.warning("Model failed to load at startup; Gradio UI will show 'model not loaded' state.")

# Hugging Face download helper (synchronous) ----------------------------------
def download_model_from_hf(repo_id: str, filename_or_pattern: Optional[str], dest_dir: str, token: Optional[str] = None) -> Optional[str]:
    """
    Attempts to download filename_or_pattern from repo_id into dest_dir.
    If filename_or_pattern is None, performs a snapshot and chooses preferred.
    Returns local path to the downloaded file or None.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except Exception as e:
        logger.error("huggingface_hub is not installed or failed to import: %s", e)
        return None

    os.makedirs(dest_dir, exist_ok=True)
    # if exact filename provided, try hf_hub_download
    if filename_or_pattern:
        try:
            logger.info("Downloading %s from %s (exact filename) ...", filename_or_pattern, repo_id)
            local = hf_hub_download(repo_id=repo_id, filename=filename_or_pattern, local_dir=dest_dir, token=token)
            logger.info("Downloaded: %s", local)
            return local
        except Exception as e:
            logger.error("hf_hub_download failed (exact filename): %s", e)

    # fallback: snapshot and find .gguf
    try:
        logger.info("Snapshotting repo %s (may take a while for large files)...", repo_id)
        snapshot_dir = snapshot_download(repo_id=repo_id, local_dir=dest_dir, token=token, allow_patterns=["*.gguf", "*.GGUF", "*"])
        # snapshot_download may return str of dir
        logger.info("Snapshot done; searching for gguf files in %s", snapshot_dir)
        found = select_local_gguf_model(snapshot_dir)
        if found:
            logger.info("Found model after snapshot: %s", found)
            return found
        else:
            logger.warning("No .gguf file found in snapshot dir: %s", snapshot_dir)
            return None
    except Exception as e:
        logger.error("snapshot_download failed: %s", e, exc_info=True)
        return None

# Build prompt helper ---------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a legal assistant specializing in Gender-Based Violence (GBV) issues across both males and females. "
    "Give clear, inclusive, neutral legal advice about GBV-related cases only. "
    "If a query is not about GBV, say you only respond to GBV cases."
)

def build_prompt(system_message: str, user_input: str) -> str:
    # Re-create format like in your original script
    return f"System: {system_message}\nUser: {user_input}\nLawyer:"

# Generation function (sync) --------------------------------------------------
def generate_sync(user_input: str, system_message: Optional[str] = None, max_tokens: int = 250, temperature: float = 0.7) -> Dict[str, Any]:
    """
    Synchronous generation wrapper used by both API and Gradio.
    Returns dict with keys: success(bool), text(str), details(dict)
    """
    if system_message is None:
        system_message = DEFAULT_SYSTEM_PROMPT
    prompt = build_prompt(system_message, user_input)

    if llm is None:
        return {"success": False, "text": "", "error": "Model not loaded in this process."}

    try:
        # llama-cpp-python typical usage: llm.create(...)
        # it returns dict with 'choices': [{'text': '...'}]
        resp = llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        text = ""
        # Some versions return 'choices' or 'response'
        if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
            # Many versions place text in resp["choices"][0]["text"]
            text = resp["choices"][0].get("text", "")
        elif isinstance(resp, dict) and "content" in resp:
            text = resp["content"]
        else:
            # Fallback to string representation
            text = str(resp)
        return {"success": True, "text": text, "details": resp}
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        return {"success": False, "text": "", "error": str(e)}

# FastAPI + Gradio startup ----------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="LLM (gguf) service with Gradio test UI")

class GenRequest(BaseModel):
    input: str
    system_message: Optional[str] = None
    max_tokens: Optional[int] = 250
    temperature: Optional[float] = 0.7

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": llm is not None, "local_model_path": LOCAL_MODEL_PATH, "model_dir": MODEL_LOCAL_PATH}

@app.post("/api/generate")
def api_generate(req: GenRequest):
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="input is required")
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded in this service.")
    out = generate_sync(req.input, req.system_message, max_tokens=req.max_tokens, temperature=req.temperature)
    if not out.get("success"):
        raise HTTPException(status_code=500, detail=out.get("error", "unknown"))
    return {"output": out["text"], "raw": out.get("details")}

@app.post("/admin/download_model")
def admin_download_model(repo_id: Optional[str] = None, filename: Optional[str] = None):
    """
    Synchronous endpoint (admin) to download model from HF into MODEL_LOCAL_PATH.
    Returns path or error. (Use HF_TOKEN env var for private repos.)
    """
    repo = repo_id or HF_REPO_ID
    if not repo:
        raise HTTPException(status_code=400, detail="repo_id not provided and HF_REPO_ID not set")
    token = HF_TOKEN
    downloaded = download_model_from_hf(repo, filename or MODEL_FILENAME, MODEL_LOCAL_PATH, token)
    if not downloaded:
        raise HTTPException(status_code=500, detail="download failed; check logs")
    # register and attempt to load
    global LOCAL_MODEL_PATH
    LOCAL_MODEL_PATH = downloaded
    load_ok = False
    if llama_backend_available:
        load_ok = load_llama_model(LOCAL_MODEL_PATH)
    return {"downloaded": downloaded, "loaded": load_ok}

# Build (simple) Gradio interface ---------------------------------------------
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except Exception:
    GRADIO_AVAILABLE = False
    logger.warning("gradio not installed; Gradio UI disabled.")

def create_gradio_demo():
    if not GRADIO_AVAILABLE:
        return None

    with gr.Blocks(title="GBV Legal Assistant (local gguf test UI)") as demo:
        gr.Markdown("## GBV legal assistant — test UI")
        model_status = gr.State({"loaded": llm is not None, "path": LOCAL_MODEL_PATH})
        with gr.Row():
            txt = gr.Textbox(lines=6, label="User input (GBV scenario or question)", placeholder="Enter a GBV-related scenario or question...")
            sys_msg = gr.Textbox(lines=2, label="System prompt (optional)", value=DEFAULT_SYSTEM_PROMPT)
        with gr.Row():
            max_t = gr.Slider(1, 2048, value=250, step=1, label="max_tokens")
            temp = gr.Slider(0.0, 1.5, value=0.7, step=0.01, label="temperature")
        out = gr.Textbox(lines=10, label="Model output")
        status = gr.Textbox(label="Model status", value=f"loaded={llm is not None}, path={LOCAL_MODEL_PATH}", interactive=False)

        def do_generate(user_input, system_prompt, max_tokens, temperature):
            if not user_input or not user_input.strip():
                return "", "Please enter user input."
            if llm is None:
                return "", "Model not loaded in process. Click 'Download & Load model' (admin) or check server logs."
            res = generate_sync(user_input, system_prompt, int(max_tokens), float(temperature))
            if not res.get("success"):
                return "", f"Error: {res.get('error')}"
            return res["text"], f"loaded={llm is not None}, path={LOCAL_MODEL_PATH}"

        gen_btn = gr.Button("Generate")
        gen_btn.click(do_generate, inputs=[txt, sys_msg, max_t, temp], outputs=[out, status])

        # Admin download button (synchronous) - downloads then tries to load
        repo_in = gr.Textbox(label="HF repo id (optional, e.g. user/repo)", value=HF_REPO_ID or "")
        fname_in = gr.Textbox(label="Filename (optional)", value=MODEL_FILENAME or "")
        download_btn = gr.Button("Download & Load model from HF")
        download_out = gr.Textbox(label="Download result")

        def download_and_load(repo_id_val, filename_val):
            repo = repo_id_val or HF_REPO_ID
            if not repo:
                return "No HF repo id provided and HF_REPO_ID env var not set."
            downloaded = download_model_from_hf(repo, filename_val or MODEL_FILENAME, MODEL_LOCAL_PATH, HF_TOKEN)
            if not downloaded:
                return "Download failed. Check server logs for details."
            # attempt to load
            if llama_backend_available:
                ok = load_llama_model(downloaded)
                return f"Downloaded to {downloaded}. load_ok={ok}"
            else:
                return f"Downloaded to {downloaded}. llama-cpp backend not available in this environment."

        download_btn.click(download_and_load, inputs=[repo_in, fname_in], outputs=[download_out])
    return demo

demo = create_gradio_demo()

# Utility: find an available port in small range ------------------------------
def is_port_free(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.close()
        return True
    except OSError:
        return False

def choose_start_port(preferred: int = DEFAULT_PORT, attempts: int = 50) -> int:
    host = "0.0.0.0"
    p = preferred
    for i in range(attempts):
        if is_port_free(host, p):
            return p
        p += 1
    logger.warning("Could not find a free port in range starting at %d", preferred)
    return preferred  # let Gradio/uvicorn raise error later

# If module is launched directly, start the Gradio demo (for Spaces/testing).
if __name__ == "__main__":
    # --- replace or update DEFAULT_PORT calculation near top of app.py ---
# prefer the platform PORT env var (Render/Heroku style); fallback to others
DEFAULT_PORT = int(os.environ.get("PORT") or os.environ.get("HF_PORT") or os.environ.get("GRADIO_SERVER_PORT") or 7861)

# --- replace the bottom "if __name__ == '__main__':" block with this ---
if __name__ == "__main__":
    start_port = DEFAULT_PORT
    logger.info("Starting service. Gradio available=%s. Using port %d", GRADIO_AVAILABLE, start_port)

    # If Gradio available and you want a dev/test UI *only* (not recommended for production):
    if GRADIO_AVAILABLE and demo is not None:
        try:
            demo.launch(
                server_name="0.0.0.0",
                server_port=start_port,
                share=False,
                prevent_thread_lock=True
            )
        except Exception as e:
            logger.exception("Failed to launch Gradio: %s. Falling back to uvicorn", e)
            import uvicorn
            uvicorn.run("app:app", host="0.0.0.0", port=start_port, log_level="info")
    else:
        # production path: run FastAPI via uvicorn
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=start_port, log_level="info")
# --- replace or update DEFAULT_PORT calculation near top of app.py ---
# prefer the platform PORT env var (Render/Heroku style); fallback to others
DEFAULT_PORT = int(os.environ.get("PORT") or os.environ.get("HF_PORT") or os.environ.get("GRADIO_SERVER_PORT") or 7861)

# --- replace the bottom "if __name__ == '__main__':" block with this ---
if __name__ == "__main__":
    start_port = DEFAULT_PORT
    logger.info("Starting service. Gradio available=%s. Using port %d", GRADIO_AVAILABLE, start_port)

    # If Gradio available and you want a dev/test UI *only* (not recommended for production):
    if GRADIO_AVAILABLE and demo is not None:
        try:
            demo.launch(
                server_name="0.0.0.0",
                server_port=start_port,
                share=False,
                prevent_thread_lock=True
            )
        except Exception as e:
            logger.exception("Failed to launch Gradio: %s. Falling back to uvicorn", e)
            import uvicorn
            uvicorn.run("app:app", host="0.0.0.0", port=start_port, log_level="info")
    else:
        # production path: run FastAPI via uvicorn
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=start_port, log_level="info")
