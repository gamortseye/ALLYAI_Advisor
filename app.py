# app.py
import os
import logging
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download, login as hf_login
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# ENV vars (set these in Render dashboard)
HF_TOKEN = os.environ.get("HF_TOKEN", None)           # REQUIRED to download private models; optional for public
MODEL_REPO = os.environ.get("MODEL_REPO", "")        # e.g. "Gamortsey/AllyAI-lawchat-gguf"
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "")# e.g. "law-chat-q4_0.gguf"
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "/app/model_files")  # where to store downloaded gguf
PORT = int(os.environ.get("PORT", "8080"))
LLAMA_THREADS = int(os.environ.get("LLAMA_THREADS", "4"))

# If HF token provided, login programmatically (so hf_hub_download works)
if HF_TOKEN:
    hf_login(HF_TOKEN)

# Create model folder
os.makedirs(MODEL_LOCAL_PATH, exist_ok=True)

# Try import llama_cpp safely
llama_available = False
llama_error = None
try:
    from llama_cpp import Llama
    llama_available = True
except Exception as e:
    llama_available = False
    llama_error = str(e)
    log.warning("llama-cpp import failed initially: %s", llama_error)

# Helper: download model from HF if missing
def ensure_model_present(repo_id: str, filename: str) -> str:
    """
    Download model file from HF repo to MODEL_LOCAL_PATH if not already present.
    Returns full local path to model file.
    """
    local_path = os.path.join(MODEL_LOCAL_PATH, filename)
    if os.path.exists(local_path):
        log.info("Model already present at %s", local_path)
        return local_path

    if not repo_id or not filename:
        raise RuntimeError("MODEL_REPO and MODEL_FILENAME must be set as environment variables if model is not bundled.")

    log.info("Downloading model %s/%s to %s ...", repo_id, filename, MODEL_LOCAL_PATH)
    # hf_hub_download will raise if not found or auth fails
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=MODEL_LOCAL_PATH, token=HF_TOKEN)
    log.info("Downloaded model to %s", local_path)
    return local_path

# Load model (if possible) and hold llama instance
llm = None
model_path = None
if MODEL_REPO and MODEL_FILENAME:
    try:
        model_path = ensure_model_present(MODEL_REPO, MODEL_FILENAME)
    except Exception as e:
        log.warning("Could not download model at startup: %s", e)
        model_path = None
else:
    log.info("MODEL_REPO or MODEL_FILENAME not set — expecting model bundled or skipping load.")

if llama_available and model_path:
    try:
        # instantiate llama-cpp Llama wrapper
        llm = Llama(model_path=model_path, n_threads=LLAMA_THREADS)
        log.info("Llama model loaded from %s", model_path)
    except Exception as e:
        log.exception("Failed to instantiate Llama with model: %s", e)
        llm = None
        llama_available = False
        llama_error = str(e)

# Define FastAPI
app = FastAPI(title="llama-cpp backend + Gradio UI")

# Simple pydantic request model
class ChatRequest(BaseModel):
    system_message: Optional[str] = "You are a legal assistant specializing in Gender-Based Violence (GBV) issues across both males and females. Be clear, inclusive, neutral. Answer only GBV-related queries."
    prompt: str
    max_tokens: Optional[int] = 250
    temperature: Optional[float] = 0.7

@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Basic synchronous completion endpoint.
    """
    if not llm:
        return {"error": "Model not loaded", "details": llama_error}

    full_prompt = f"{req.system_message}\n\nUser: {req.prompt}\nLawyer:"
    try:
        r = llm.create(prompt=full_prompt, max_tokens=req.max_tokens, temperature=req.temperature)
        # llm.create returns a dict with 'choices' typically
        text = ""
        if isinstance(r, dict) and "choices" in r:
            # join all choices text
            text = "".join([c.get("text", "") for c in r["choices"]])
        else:
            # fallback: try to stringify
            text = str(r)
        return {"generated_text": text, "raw": r}
    except Exception as e:
        log.exception("Generation failed: %s", e)
        return {"error": "generation_failed", "details": str(e)}

# If llama isn't available, a lightweight health endpoint
@app.get("/health")
def health():
    return {
        "llama_cpp_available": llama_available,
        "import_error": llama_error,
        "model_loaded": bool(llm),
        "model_path": model_path,
    }

# ---------------------
# Gradio UI for testing (mounted into the same process)
# ---------------------
import gradio as gr
def generate_for_ui(user_prompt, system_prompt, max_tokens, temperature):
    req = ChatRequest(system_message=system_prompt, prompt=user_prompt, max_tokens=int(max_tokens), temperature=float(temperature))
    out = chat(req)
    if "generated_text" in out:
        return out["generated_text"]
    else:
        return f"Error: {out.get('error')}\nDetails: {out.get('details')}"

with gr.Blocks() as demo:
    gr.Markdown("# Llama-cpp law-chat (GBV assistant) — Gradio UI")
    system_tb = gr.Textbox(label="System prompt", value="You are a legal assistant specializing in Gender-Based Violence (GBV) issues across both males and females. Be clear, inclusive, neutral. Answer only GBV-related queries.")
    user_tb = gr.Textbox(label="User prompt", lines=6, placeholder="Describe the case...")
    tokens = gr.Slider(10, 1024, value=250, label="max tokens")
    temp = gr.Slider(0.0, 1.2, value=0.7, label="temperature")
    btn = gr.Button("Generate")
    out_tb = gr.Textbox(label="Response", lines=10)
    btn.click(fn=generate_for_ui, inputs=[user_tb, system_tb, tokens, temp], outputs=[out_tb])

# Mount Gradio on FastAPI at root path
from fastapi import Request
app.mount("/", gr.mount_gradio_app(app, demo, path="/"))

# Note: Render will run this file via gunicorn + uvicorn worker configured in Dockerfile.
