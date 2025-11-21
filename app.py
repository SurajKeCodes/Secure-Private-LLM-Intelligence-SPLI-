# == [1] Imports and Constants ==
import streamlit as st
import os
import requests
from llama_cpp import Llama
from io import StringIO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from vosk import Model as VoskModel, KaldiRecognizer
import wave
import json
import subprocess

# == GGUF Models ==
MODELS = {
    "TinyLLaMA 1.1B Q4_K_M": {
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1b-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-Q4_K_M.gguf"
    },
    "TinyLLaMA 1.1B Q5_K_S": {
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1b-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_S.gguf",
        "filename": "tinyllama-Q5_K_S.gguf"
    },
    "Phi-2 Q4_K_M": {
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "filename": "phi2-Q4_K_M.gguf"
    },
    "Mistral 7B Q4_K_M": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "filename": "mistral-Q4_K_M.gguf"
    },
    "LLaMA2 7B Q4_K_S": {
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_S.gguf",
        "filename": "llama2-Q4_K_S.gguf"
    }
}

MODEL_DIR = "models"
VOSK_MODEL_PATH = "vosk-model/vosk-model-small-en-us-0.15"
os.makedirs(MODEL_DIR, exist_ok=True)

# == [2] Helper Functions ==
@st.cache_resource
def load_vosk_model():
    return VoskModel(VOSK_MODEL_PATH)

def has_cuda():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

@st.cache_resource
def load_llm(path, use_gpu):
    try:
        if use_gpu:
            st.sidebar.info("⚡ Loading model with GPU...")
            return Llama(model_path=path, n_ctx=2048, n_threads=6, n_gpu_layers=35)
        else:
            st.sidebar.info("🧠 Loading model on CPU...")
            return Llama(model_path=path, n_ctx=2048, n_threads=6)
    except Exception as e:
        st.sidebar.error(f"❌ Model load failed: {e}")
        return None

def download_model(url, filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path) and os.path.getsize(path) > 10 * 1024 * 1024:
        st.sidebar.success("✅ Model already downloaded.")
        return path
    try:
        with requests.get(url, stream=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            progress = st.sidebar.progress(0, "⏬ Downloading model...")
            with open(path, "wb") as f:
                for chunk in resp.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = downloaded / total
                        progress.progress(percent, f"⏬ Downloading... {int(percent * 100)}%")
        st.sidebar.success("✅ Download complete.")
        return path
    except Exception as e:
        st.sidebar.error(f"❌ Download failed: {e}")
        return None

# == [3] Streamlit State Setup ==
st.session_state.setdefault("model_path", None)
st.session_state.setdefault("llm", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("system_prompt", "You are a helpful assistant.")
st.session_state.setdefault("last_user_input", None)
st.session_state.setdefault("personas", {
    "Default": "You are a helpful assistant.",
    "Friendly Teacher": "You explain things in a friendly and patient manner.",
    "Grumpy Developer": "You're helpful but a bit sarcastic and grumpy.",
    "Legal Advisor": "You give precise and legally safe advice."
})

# == [4] UI Rendering ==
st.title("🧠 Local LLM Chat with Voice")
st.sidebar.header("🧩 Model Selection")
choice = st.sidebar.selectbox("Choose a model", [""] + list(MODELS.keys()))
upload = st.sidebar.file_uploader("Or upload .gguf model", type="gguf")
default_gpu = has_cuda()
use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=default_gpu)

if upload:
    path = os.path.join(MODEL_DIR, upload.name)
    with open(path, "wb") as f:
        f.write(upload.read())
    st.session_state.model_path = path
    st.session_state.llm = load_llm(path, use_gpu)
elif choice:
    info = MODELS[choice]
    if st.sidebar.button("⬇️ Download and Load"):
        path = download_model(info["url"], info["filename"])
        if path:
            st.session_state.model_path = path
            st.session_state.llm = load_llm(path, use_gpu)

st.sidebar.markdown("### 🔧 Model Settings")
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
max_tokens = st.sidebar.slider("Max tokens", 100, 2048, 512)

st.sidebar.markdown("### 🧠 LLM Behavior Settings")
persona_choice = st.sidebar.selectbox("Choose Assistant Persona", list(st.session_state.personas.keys()))
st.session_state.system_prompt = st.session_state.personas[persona_choice]
custom_prompt = st.sidebar.text_area("✏️ Custom System Prompt (overrides persona)", value=st.session_state.system_prompt)
if custom_prompt != st.session_state.system_prompt:
    st.session_state.system_prompt = custom_prompt

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history.clear()

if st.sidebar.button("💾 Export Chat (.txt)") and st.session_state.history:
    chat_text = "\n\n".join(f"User: {u}\nAssistant: {a}" for u, a in st.session_state.history)
    chat_file = StringIO(chat_text)
    st.sidebar.download_button("Download", chat_file, file_name="chat_history.txt")

# == [5] Chat Logic ==
llm = st.session_state.llm
if llm:
    for i, (user_msg, bot_msg) in enumerate(st.session_state.history):
        st.chat_message("user").markdown(user_msg)
        st.chat_message("assistant").markdown(bot_msg)

        if st.button("🔄 Regenerate", key=f"regen_{i}"):
            full_prompt = f"<<SYS>>\n{st.session_state.system_prompt}\n<</SYS>>\n\n"
            for j, (u, a) in enumerate(st.session_state.history[-4:]):
                if j == i:
                    full_prompt += f"[INST] {u} [/INST]"
                    break
                full_prompt += f"[INST] {u} [/INST] {a}\n"
            result = llm(full_prompt, max_tokens=max_tokens, temperature=temperature, stop=["</s>"])
            new_answer = result["choices"][0]["text"].strip()
            st.session_state.history[i] = (user_msg, new_answer)
            st.experimental_rerun()

    user_input = st.chat_input("💬 Type your question or use microphone below...")

    with st.expander("🎤 Record from Microphone"):
        try:
            webrtc_streamer(
                key="mic",
                mode=WebRtcMode.SENDRECV,
                audio_receiver_size=1024,
                media_stream_constraints={"audio": True, "video": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                sendback_audio=False,
            )
            st.caption("🎙️ Real-time mic capture enabled (transcription not yet live).")
        except KeyError:
            st.warning("⚠️ Microphone stream initializing, please wait or reload.")

    audio_file = st.file_uploader("🎙 Upload voice (.wav, mono PCM)", type=["wav"])
    if audio_file and not user_input:
        with wave.open(audio_file, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                st.error("Audio must be WAV format with mono PCM encoding.")
            else:
                st.info("🧠 Transcribing with Vosk...")
                model = load_vosk_model()
                rec = KaldiRecognizer(model, wf.getframerate())
                rec.SetWords(True)

                result = ""
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        part_result = json.loads(rec.Result())
                        result += part_result.get("text", "") + " "

                final_result = json.loads(rec.FinalResult())
                result += final_result.get("text", "")
                user_input = result.strip()

    if user_input and user_input != st.session_state.last_user_input:
        st.session_state.last_user_input = user_input
        st.chat_message("user").markdown(user_input)

        full_prompt = f"<<SYS>>\n{st.session_state.system_prompt}\n<</SYS>>\n\n"
        for u, a in st.session_state.history[-4:]:
            full_prompt += f"[INST] {u} [/INST] {a}\n"
        full_prompt += f"[INST] {user_input} [/INST]"

        try:
            with st.spinner("🤔 Thinking..."):
                result = llm(full_prompt, max_tokens=max_tokens, temperature=temperature, stop=["</s>"])
                answer = result["choices"][0]["text"].strip()
        except Exception as e:
            answer = f"❌ Error: {e}"

        st.chat_message("assistant").markdown(answer)
        st.session_state.history.append((user_input, answer))
        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[-20:]
else:
    st.info("📂 Please upload or download a model to begin chatting.")

# == [6] Footer ==
st.markdown("""
<hr>
<div style='text-align: center;'>
    <p>Made with ❤️ by <strong>SS INFOTECH PVT LTD</strong></p>
    <a href='https://www.ssinfotech.co/' target='_blank'>
        <button style='padding: 0.5em 1em; font-size: 1em;'>Visit Site</button>
    </a>
</div>
""", unsafe_allow_html=True)
