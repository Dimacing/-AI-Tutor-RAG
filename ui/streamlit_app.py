import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Tutor RAG")
st.title("AI Tutor RAG")

PROVIDERS = ["openai", "gemini", "deepseek", "ollama", "gigachat"]
MODEL_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "deepseek": "deepseek-chat",
    "ollama": "llama3.1",
    "gigachat": "GigaChat",
}
MODE_OPTIONS = {
    "Cloud docs": "cloud",
    "My files": "local",
    "Website": "web",
}


def fetch_health() -> dict:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
    except requests.RequestException:
        return {}
    if response.status_code != 200:
        return {}
    return response.json()


def fetch_embedding_models() -> tuple[list[str], str]:
    fallback = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]
    try:
        response = requests.get(f"{BACKEND_URL}/embedding-models", timeout=10)
    except requests.RequestException:
        return fallback, fallback[0]
    if response.status_code != 200:
        return fallback, fallback[0]
    payload = response.json()
    models = payload.get("models") or fallback
    default = payload.get("default") or models[0] or fallback[0]
    return models, default

if "provider" not in st.session_state:
    st.session_state.provider = "openai"
if "model" not in st.session_state:
    st.session_state.model = MODEL_DEFAULTS[st.session_state.provider]
if "provider_last" not in st.session_state:
    st.session_state.provider_last = st.session_state.provider
if "mode_label" not in st.session_state:
    st.session_state.mode_label = "Cloud docs"
if "mode" not in st.session_state:
    st.session_state.mode = MODE_OPTIONS[st.session_state.mode_label]
if "cloud_index_confirmed" not in st.session_state:
    st.session_state.cloud_index_confirmed = False

health = fetch_health()
embedding_models, embedding_default = fetch_embedding_models()
cloud_ready = bool(health.get("index_ready"))
local_ready = bool(health.get("local_index_ready"))
web_ready = bool(health.get("web_index_ready"))

with st.sidebar:
    mode_label = st.radio("Mode", list(MODE_OPTIONS.keys()), key="mode_label")
    mode = MODE_OPTIONS[mode_label]
    st.session_state.mode = mode

    if mode == "cloud":
        st.caption(f"Cloud index: {'ready' if cloud_ready else 'not ready'}")
        if cloud_ready and not st.session_state.cloud_index_confirmed:
            st.info("Click 'Build cloud index' to use this mode.")
        if st.button("Build cloud index"):
            with st.spinner("Indexing cloud docs..."):
                response = requests.post(
                    f"{BACKEND_URL}/cloud-index",
                    json={"embedding_model": embedding_model},
                    timeout=600,
                )
            if response.status_code != 200:
                st.error(response.text)
            else:
                payload = response.json()
                st.success(
                    "Indexed "
                    f"{payload.get('num_documents', 0)} documents into "
                    f"{payload.get('num_chunks', 0)} chunks."
                )
                cloud_ready = True
                st.session_state.cloud_index_confirmed = True

    if mode == "local":
        st.caption(f"Local index: {'ready' if local_ready else 'not ready'}")
        uploaded_files = st.file_uploader(
            "Upload files (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        if st.button("Build local index"):
            if not uploaded_files:
                st.warning("Upload at least one file.")
            else:
                files_payload = []
                for uploaded in uploaded_files:
                    files_payload.append(
                        (
                            "files",
                            (
                                uploaded.name,
                                uploaded.getvalue(),
                                uploaded.type or "application/octet-stream",
                            ),
                        )
                    )
                with st.spinner("Indexing local files..."):
                    response = requests.post(
                        f"{BACKEND_URL}/local-index",
                        files=files_payload,
                        params={"embedding_model": embedding_model},
                        timeout=600,
                    )
                if response.status_code != 200:
                    st.error(response.text)
                else:
                    payload = response.json()
                    st.success(
                        "Indexed "
                        f"{payload.get('num_files', 0)} files into "
                        f"{payload.get('num_chunks', 0)} chunks."
                    )
                    local_ready = True

    if mode == "web":
        st.caption(f"Web index: {'ready' if web_ready else 'not ready'}")
        web_url = st.text_input("Website URL", key="web_url")
        crawl = st.checkbox("Crawl links on the same domain", value=True, key="web_crawl")
        allowlist = st.text_input("Allowlist path (optional)", key="web_allowlist")
        max_pages = st.number_input(
            "Max pages",
            min_value=1,
            max_value=200,
            value=30,
            step=1,
            key="web_max_pages",
        )
        if st.button("Build website index"):
            if not web_url.strip():
                st.warning("Enter a website URL first.")
            else:
                payload = {
                    "url": web_url.strip(),
                    "crawl": crawl,
                    "max_pages": max_pages,
                    "embedding_model": embedding_model,
                }
                if allowlist.strip():
                    payload["allowlist"] = allowlist.strip()
                with st.spinner("Indexing website..."):
                    response = requests.post(
                        f"{BACKEND_URL}/web-index",
                        json=payload,
                        timeout=600,
                    )
                if response.status_code != 200:
                    st.error(response.text)
                else:
                    payload = response.json()
                    st.success(
                        "Indexed "
                        f"{payload.get('num_documents', 0)} documents into "
                        f"{payload.get('num_chunks', 0)} chunks."
                    )
                    web_ready = True

    provider = st.selectbox("Provider", PROVIDERS, key="provider")
    if provider != st.session_state.provider_last:
        st.session_state.model = MODEL_DEFAULTS.get(provider, "")
        st.session_state.provider_last = provider
    model = st.text_input("Model", key="model")
    embedding_model = st.selectbox(
        "Embedding model",
        embedding_models,
        index=embedding_models.index(embedding_default)
        if embedding_default in embedding_models
        else 0,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

mode = st.session_state.get("mode", "cloud")
mode_ready = (
    (cloud_ready and st.session_state.cloud_index_confirmed)
    if mode == "cloud"
    else local_ready if mode == "local" else web_ready
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not mode_ready:
    if mode == "cloud" and cloud_ready and not st.session_state.cloud_index_confirmed:
        st.warning("Cloud docs require a fresh build. Click 'Build cloud index' in the sidebar.")
    else:
        st.warning("Index is not ready. Build embeddings in the sidebar.")

prompt = st.chat_input(
    "Ask a question about the learning materials",
    disabled=not mode_ready,
)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={
                    "question": prompt,
                    "mode": mode,
                    "provider": provider,
                    "model": model,
                },
                timeout=120,
            )

        answer = ""
        if response.status_code != 200:
            error_text = response.text.strip() or f"HTTP {response.status_code}"
            st.error(error_text)
            answer = f"Error: {error_text}"
        else:
            payload = response.json()
            answer = payload.get("answer", "")
            self_check = payload.get("self_check_question", "")
            resources = payload.get("recommended_resources", [])

            st.markdown(answer)
            if self_check:
                st.markdown(f"**Self-check question:** {self_check}")
            if resources:
                st.markdown("**Recommended resources:**")
                for item in resources:
                    title = item.get("title", "resource")
                    url = item.get("url", "")
                    st.markdown(f"- {title} - {url}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
