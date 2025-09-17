# frontend/streamlit_app.py
import os
import time
from io import BytesIO

import requests
import streamlit as st

APP_NAME = "DocSage"
APP_TAGLINE = "Ask. Retrieve. Understand. — Source-backed Document Q&A"
ACCENT = "#0f4c81"
BACKEND_DEFAULT = "http://127.0.0.1:8000"

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(f"🧠 {APP_NAME}")
st.caption(APP_TAGLINE)

if "history" not in st.session_state:
    st.session_state.history = []
if "docs" not in st.session_state:
    st.session_state.docs = []
if "backend_url" not in st.session_state:
    st.session_state.backend_url = BACKEND_DEFAULT

# Sidebar
with st.sidebar:
    st.header("1) Upload & Ingest")
    uploaded = st.file_uploader("Upload files", type=["pdf","docx","pptx","txt","csv","png","jpg","jpeg"], accept_multiple_files=True)
    domain_choice = st.selectbox("Document domain", ["general","pharma","legal","engineering"], index=0)
    tags_input = st.text_input("Tags (comma separated)", value="")
    ingest_btn = st.button("Ingest files")
    st.markdown("---")
    st.header("2) Settings")
    st.session_state.backend_url = st.text_input("Backend URL", value=st.session_state.backend_url)
    top_k_sidebar = st.slider("Top K (retrieval)", 1, 10, 5)
    adaptive_k_sidebar = st.checkbox("Adaptive K (experimental)", value=False)

# Ingest logic
if ingest_btn:
    if not uploaded:
        st.sidebar.warning("Please upload files to ingest.")
    else:
        with st.spinner("Uploading & ingesting..."):
            prog = st.progress(0)
            for i, f in enumerate(uploaded):
                files = {"file": (f.name, f.getvalue(), "application/octet-stream")}
                data = {"domain": domain_choice, "tags": tags_input}
                try:
                    r = requests.post(f"{st.session_state.backend_url.rstrip('/')}/ingest/", files=files, data=data, timeout=300)
                    if r.status_code == 200:
                        j = r.json()
                        st.sidebar.success(f"Ingested {f.name} — chunks: {j.get('chunks')}")
                        if f.name not in st.session_state.docs:
                            st.session_state.docs.append(f.name)
                    else:
                        st.sidebar.error(f"Failed {f.name}: {r.text}")
                except Exception as e:
                    st.sidebar.error(f"Error {f.name}: {e}")
                prog.progress(int((i+1)/len(uploaded)*100))
            prog.empty()
        st.experimental_rerun()

# Main UI
col_main, col_meta = st.columns([3,1])
with col_main:
    st.header("Ask DocSage")
    question = st.text_input("Your question about the ingested documents")
    k_val = st.number_input("Top K (override)", min_value=1, max_value=10, value=top_k_sidebar)
    adaptive_k = st.checkbox("Adaptive K (send to backend)", value=adaptive_k_sidebar)
    domain_for_query = st.selectbox("Domain for query (influence prompt)", ["general","pharma","legal","engineering"], index=0)

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            payload = {"question": question, "top_k": int(k_val), "adaptive_k": bool(adaptive_k), "domain": domain_for_query}
            with st.spinner("Querying backend..."):
                try:
                    r = requests.post(f"{st.session_state.backend_url.rstrip('/')}/query/", json=payload, timeout=300)
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    r = None
                if r and r.status_code == 200:
                    resp = r.json()
                    answer = resp.get("answer","")
                    citations = resp.get("citations",[])
                    st.subheader("Answer")
                    st.write(answer)
                    st.download_button("Download answer (.txt)", data=answer, file_name="answer.txt")
                    st.subheader("Retrieved citations")
                    if not citations:
                        st.info("No citations returned.")
                    for i, c in enumerate(citations):
                        with st.expander(f"{i+1}. {c.get('source','unknown')} (chunk {c.get('chunk_index',-1)})", expanded=(i==0)):
                            st.markdown(f"**Source:** {c.get('source','unknown')}")
                            st.markdown(f"**Chunk index:** {c.get('chunk_index',-1)}")
                            st.write(c.get("text",""))
                            local_path = os.path.abspath(os.path.join(os.getcwd(), "..", "data", c.get("source","")))
                            if os.path.exists(local_path):
                                st.markdown(f"_Local file:_ `{local_path}`")
                    # Save to history
                    st.session_state.history.insert(0, {"q": question, "a": answer, "citations": citations})
                else:
                    st.error(f"Backend error: {r.status_code if r else 'N/A'} — {r.text if r else ''}")

    st.markdown("### Conversation history")
    for idx, h in enumerate(st.session_state.history):
        st.markdown(f"**Q:** {h['q']}")
        st.markdown(f"**A:** {h['a']}")
        if st.button("Remove", key=f"rm_{idx}"):
            st.session_state.history.pop(idx)
            st.experimental_rerun()

with col_meta:
    st.markdown("### Ingested files")
    if st.session_state.docs:
        for fn in st.session_state.docs:
            st.markdown(f"- {fn}")
    else:
        st.markdown("_No documents ingested_")
    st.markdown("---")
    st.markdown("**Backend**")
    st.write(st.session_state.backend_url)
    if st.button("Open API docs"):
        st.markdown(f"[Open Swagger]({st.session_state.backend_url.rstrip('/')}/docs)")

st.markdown("---")
st.caption("DocSage — local RAG demo. Change domain during ingest/query to influence response style.")
