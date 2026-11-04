"""Streamlit UI for MedRAG-DE. Talks to the FastAPI service at API_URL."""

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))
TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "60")) + 5
MODES = ["hybrid_rerank", "hybrid", "dense", "bm25"]
EXAMPLES = [
    "Ab welchem Blutdruckwert sollte eine medikamentöse Behandlung begonnen werden?",
    "How much physical activity is recommended for adults?",
]

st.set_page_config(page_title="MedRAG-DE", page_icon="🩺")
st.title("MedRAG-DE")
st.caption("Local, bilingual (DE/EN) question answering over public health guidelines.")
st.warning(
    "**Demo only – not medical advice.** Answers can be wrong or incomplete. "
    "Always check the cited source. / **Nur Demo – keine medizinische Beratung.**"
)

with st.sidebar:
    mode = st.selectbox("Retrieval mode", MODES, help="hybrid_rerank is usually the most precise.")
    top_k = st.slider("Sources (top_k)", min_value=1, max_value=10, value=5)
    st.markdown(f"API: `{API_URL}`")

question = st.text_area(
    "Question / Frage",
    placeholder=EXAMPLES[0],
    max_chars=MAX_QUESTION_CHARS,
    height=100,
)

if st.button("Ask / Fragen", type="primary", disabled=not question.strip()):
    with st.spinner("Searching the documents…"):
        try:
            response = httpx.post(
                f"{API_URL}/query",
                json={"question": question, "mode": mode, "top_k": top_k},
                timeout=TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the API ({exc.__class__.__name__}). Is it running?")
            st.stop()

    if response.status_code != 200:
        st.error(f"API error {response.status_code}: {response.json().get('detail')}")
        st.stop()

    result = response.json()
    if result["found"]:
        st.success(result["answer"])
    else:
        st.info(result["answer"])

    latency = result["latency"]
    left, middle, right = st.columns(3)
    left.metric("Retrieval", f"{latency['retrieval_ms']:.0f} ms")
    middle.metric("Generation", f"{latency['generation_ms']:.0f} ms")
    right.metric("Total", f"{latency['total_ms']:.0f} ms")

    if result["citations"]:
        st.subheader("Sources")
        for citation in result["citations"]:
            with st.expander(f"{citation['title']}, p. {citation['page']}"):
                st.write(citation["snippet"])
                st.caption(f"score {citation['score']} · {citation['doc_id']}")
