import json, os, io
import streamlit as st
import hashlib
import tempfile
import base64

from dotenv import load_dotenv
from openai import OpenAI
from backend.rag import search_books
from backend.tools import get_summary_by_title

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Smart Librarian", page_icon="📚")
st.title("📚 Smart Librarian — RAG + Tools")

# ---- Session-state init (must run before any use) ----
def init_state():
    st.session_state.setdefault("chat_text", None)     # final text shown to user (recommendation + summary)
    st.session_state.setdefault("final_title", None)   # chosen title (for filename)
    st.session_state.setdefault("last_query", None)

    st.session_state.setdefault("tts_audio", None)     # audio bytes
    st.session_state.setdefault("tts_fmt", "mp3")
    st.session_state.setdefault("tts_title", None)
    st.session_state.setdefault("voice_select", "alloy")
    st.session_state.setdefault("fmt_select", "mp3")
    st.session_state.setdefault("tts_force_open", False)  # keep expander open across reruns

    # NEW: voice/STT
    st.session_state.setdefault("voice_transcript", None)
    st.session_state.setdefault("prefill_query", "")   # to prefill the text box from transcript


    st.session_state.setdefault("cover_image", None)      # bytes (PNG)
    st.session_state.setdefault("cover_prompt", "")       # last used prompt
    st.session_state.setdefault("cover_force_open", False)
init_state()


# simple, optional profanity gate (don’t call LLM if triggered)
BAD_WORDS = {"idiot","stupid","hate"}  # expand as needed

@st.cache_data
def get_tool_spec():
    return [{
    "type": "function",
    "name": "get_summary_by_title",           # <-- name at top-level
    "description": "Return the full summary for an exact book title.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Exact book title"}
        },
        "required": ["title"]
    }
}]

def call_model_for_reco(user_query: str, top_hit_title: str):
    system = (
        "You are a helpful librarian. Recommend a single best-matching title from "
        "the provided candidate and explain briefly why it fits the user's interests."
    )
    # You can pass candidate context inline to the model:
    user = f"User wants: {user_query}\nCandidate: {top_hit_title}\nRespond with the chosen title on the first line."
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role":"system","content":system},{"role":"user","content":user}],
        tools=get_tool_spec(),
        tool_choice="auto"
    )
    return resp

def handle_tool_calls(resp):
    # Find a tool call for get_summary_by_title and execute locally
    out_text = []
    chosen_title = None

    for item in resp.output or []:
        if item.type == "message" and item.role == "assistant":
            out_text.append("".join([p.text for p in item.content if getattr(p, 'type', None) == 'output_text']))
        if item.type == "tool_call" and item.tool_name == "get_summary_by_title":
            args = item.arguments or {}
            chosen_title = args.get("title")
            summary = get_summary_by_title(chosen_title) if chosen_title else "Summary not found."
            out_text.append(f"\n\n**Full summary — {chosen_title}:**\n{summary}")

    return "\n".join(t for t in out_text if t), chosen_title


@st.cache_data(show_spinner=False)
def synthesize_tts_bytes(text: str, voice: str = "alloy", fmt: str = "mp3") -> bytes:
    _ = hashlib.sha256((voice + fmt + text).encode("utf-8")).hexdigest()
    resp = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        response_format=fmt
    )
    if hasattr(resp, "read") and callable(resp.read):
        data = resp.read()
    elif hasattr(resp, "content"):
        data = resp.content
    else:
        data = resp
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    raise RuntimeError("TTS response not in a known bytes format.")


def transcribe_uploaded_audio(uploaded, model: str = "gpt-4o-mini-transcribe") -> str:
    """
    uploaded: the value returned by st.audio_input (UploadedFile-like)
    Returns transcript text.
    """
    # Persist to a temp file so the SDK can stream it reliably
    suffix = ".wav"  # st.audio_input usually returns WAV; OpenAI accepts many formats
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    with open(tmp_path, "rb") as fh:
        # Audio → Transcriptions endpoint
        # Models: gpt-4o-mini-transcribe / gpt-4o-transcribe / whisper-1
        # (All accept file + return .text)
        tr = client.audio.transcriptions.create(
            model=model,
            file=fh
        )
    # Most SDK responses expose .text for the transcription
    return getattr(tr, "text", str(tr))

def generate_cover_image(prompt: str, size: str = "1024x1024") -> bytes:
    """
    Uses OpenAI Images API (gpt-image-1) to create a PNG image from text.
    Returns raw PNG bytes.
    """
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,
        quality="high",     # optional; supported by Images API
        n=1
    )
    b64 = result.data[0].b64_json
    return base64.b64decode(b64)


with st.expander("🎙️ Voice mode (Speech → Text)", expanded=False):
    mic = st.audio_input("Hold to record, then release")
    st.caption("Tip: speak clearly; pause a moment after releasing to attach the recording.")

    # Transcribe button appears once we have audio
    if mic is not None:
        st.audio(mic)  # playback preview
        colA, colB = st.columns([1, 1])
        with colA:
            st.write("Model:")
            stt_model = st.selectbox(
                "Transcription model",
                ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"],
                index=0,
                label_visibility="collapsed",
                key="stt_model_select"
            )
        with colB:
            run_now = st.checkbox("Use transcript as query immediately", value=True)

        if st.button("📝 Transcribe", key="transcribe_btn"):
            with st.spinner("Transcribing..."):
                try:
                    txt = transcribe_uploaded_audio(mic, model=stt_model)
                    st.session_state["voice_transcript"] = txt or ""
                    if run_now:
                        # Prefill the chat text box and auto-run in the next section
                        st.session_state["prefill_query"] = st.session_state["voice_transcript"]
                        st.toast("Transcript captured. Ready to search 🔎", icon="✅")
                    else:
                        st.toast("Transcript ready ✨", icon="✅")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")

    # Show transcript if we have one (even across reruns)
    if st.session_state.get("voice_transcript"):
        st.text_area("Transcript", st.session_state["voice_transcript"], height=120)
        if st.button("Use transcript as query", key="use_transcript_btn"):
            st.session_state["prefill_query"] = st.session_state["voice_transcript"]
            st.rerun()


with st.form("chat"):
    query = st.text_input(
        "Ask for a book by theme/keywords (e.g., “friendship and magic”, “war stories”)",
        value=st.session_state.get("prefill_query", ""),
        key="query_text"
    )
    submitted = st.form_submit_button("Recommend")

if submitted:
    if not query.strip():
        st.warning("⚠️ Please enter a theme, keyword, or question before submitting.")
    else:
        # Clear prefill & reset state
        st.session_state["prefill_query"] = ""
        st.session_state["tts_audio"] = None
        st.session_state["tts_title"] = None
        st.session_state["tts_force_open"] = False
        st.session_state["cover_image"] = None
        st.session_state["cover_prompt"] = ""
        st.session_state["cover_force_open"] = False

        # … your existing retrieval + LLM flow …
        hits = search_books(query, k=3)
        if not hits:
            st.session_state["chat_text"] = "No matches found."
            st.session_state["final_title"] = None
        else:
            top = hits[0]["title"]
            resp = call_model_for_reco(query, top)
            text, final_title = handle_tool_calls(resp)
            st.session_state["chat_text"] = text or ""
            st.session_state["final_title"] = final_title or top
            st.session_state["last_query"] = query



# Render whatever we have from session_state (survives reruns)
if st.session_state["chat_text"]:
    st.markdown(st.session_state["chat_text"])
else:
    st.info("Ask for a book above to get a recommendation and summary.")


# Auto-open expander if we have audio or we are in a generation flow
exp_open = bool(st.session_state.get("tts_audio") or st.session_state.get("tts_force_open"))

with st.expander("🔊 Read it to me", expanded=exp_open):
    col1, col2 = st.columns([2, 1])
    with col1:
        voice = st.selectbox("Voice", ["alloy", "echo", "verse"], index=0, key="voice_select")

    # Show player if audio exists; otherwise show Generate button
    if st.session_state["tts_audio"] is not None:
        st.audio(st.session_state["tts_audio"], format=f"audio/{st.session_state['tts_fmt']}")
        if st.button("Clear audio"):
            st.session_state["tts_audio"] = None
            st.session_state["tts_title"] = None
            st.session_state["tts_force_open"] = False
            st.rerun()
    else:
        # Only allow generation if we have text to read
        disabled = not bool(st.session_state["chat_text"])
        if st.button("Generate audio", key="tts_button", disabled=disabled):
            st.session_state["tts_force_open"] = True  # keep expander open across reruns
            try:
                with st.spinner("Generating audio..."):
                    audio = synthesize_tts_bytes(
                        st.session_state["chat_text"],
                        voice=st.session_state["voice_select"],
                        fmt=st.session_state["fmt_select"]
                    )
                st.session_state["tts_audio"] = audio
                st.session_state["tts_fmt"] = st.session_state["fmt_select"]
                st.session_state["tts_title"] = (st.session_state["final_title"] or "book")
                st.toast("Audio ready ✅", icon="✅")
                st.rerun()  # now the player shows and the button disappears
            except Exception as e:
                st.error(str(e))

# ---------- Cover Image UI ----------
exp_open_img = bool(st.session_state.get("cover_image") or st.session_state.get("cover_force_open"))

with st.expander("🎨 Generate cover image", expanded=exp_open_img):
    # Build a sensible default prompt from the chosen title + the visible summary text
    default_title = st.session_state.get("final_title") or "the recommended book"
    default_prompt = (
        f"Design a clean, modern **book cover** illustration for '{default_title}'. "
        "Minimalistic, high-contrast, readable at thumbnail size. No real author names or logos. "
        "Use symbolic imagery that reflects the book’s core themes from this description:\n\n"
        f"{(st.session_state.get('chat_text') or '')[:800]}"
    )
    # Let the user edit the prompt if they want
    prompt = st.text_area(
        "Image prompt",
        value=st.session_state.get("cover_prompt") or default_prompt,
        height=160
    )

    colA, colB = st.columns([1,1])
    with colA:
        size = st.selectbox("Size", ["1024x1024", "512x512", "256x256"], index=0)
    with colB:
        style_hint = st.selectbox("Style hint (optional)", ["None", "Graphic", "Watercolor", "Oil paint", "Flat vector"], index=0)

    # If image exists → show it and hide the button
    if st.session_state.get("cover_image") is not None:
        st.image(st.session_state["cover_image"], caption=f"Cover concept — {default_title}", use_container_width=True)
        st.download_button("Download PNG", st.session_state["cover_image"], file_name=f"{default_title}-cover.png")
        if st.button("Clear image"):
            st.session_state["cover_image"] = None
            st.session_state["cover_prompt"] = ""
            st.session_state["cover_force_open"] = False
            st.rerun()
    else:
        # Show the button only when no image yet and we have text worth illustrating
        disabled = not bool(st.session_state.get("chat_text"))
        if st.button("Generate cover image", disabled=disabled):
            st.session_state["cover_force_open"] = True
            # light style nudge (optional)
            final_prompt = prompt if style_hint == "None" else f"{prompt}\n\nStyle: {style_hint.lower()}."
            try:
                with st.spinner("Generating image..."):
                    img_bytes = generate_cover_image(final_prompt, size=size)
                st.session_state["cover_image"] = img_bytes
                st.session_state["cover_prompt"] = final_prompt
                st.toast("Cover image ready ✅", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"Image generation failed: {e}")

st.caption("Powered by OpenAI (tools/function calling, embeddings) + ChromaDB + Streamlit.")
