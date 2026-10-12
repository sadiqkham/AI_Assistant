import os
import hashlib
import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="IBS PDF Assistant", page_icon="📚", layout="centered")

st.title("📚 IBS Document Assistant")
st.write("Upload a department timetable or another PDF, then ask questions about information inside it.")

# Configure secrets in Streamlit Cloud: App settings → Secrets
def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY", "")

@st.cache_resource(show_spinner="Loading sentence-transformer model...")
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_pdf_pages(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": page_num, "text": text})
    return pages

def chunk_pages(pages, chunk_size=850, overlap=150):
    chunks = []
    for item in pages:
        text = " ".join(item["text"].split())
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": item["page"],
                })
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks

def build_faiss_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity for normalized vectors
    index.add(vectors)
    return index

def retrieve_chunks(question, chunks, index, model, top_k=5):
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")
    k = min(top_k, len(chunks))
    scores, ids = index.search(query_vector, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            result = dict(chunks[int(idx)])
            result["score"] = float(score)
            results.append(result)
    return results

def answer_question(question, retrieved_chunks, api_key, model_name):
    client = Groq(api_key=api_key)
    context = "\n\n".join(
        f"[PDF page {item['page']}]\n{item['text']}"
        for item in retrieved_chunks
    )
    system_prompt = """
You are an accurate assistant that answers questions using only the supplied PDF excerpts.
Rules:
1. Use only the provided context. Do not use outside knowledge to fill gaps.
2. If the answer is not clearly present, say: "I couldn't find this information in the uploaded PDF."
3. For timetable questions, carefully identify teacher, subject, day, time, class/section, and room.
4. Do not guess unclear timetable cells. If a detail is ambiguous, say it is unclear in the PDF.
5. Keep the answer direct and easy to understand.
6. Mention the PDF page number(s) supporting the answer.
"""
    user_prompt = f"""PDF EXCERPTS:
{context}

USER QUESTION:
{question}

Answer using the rules above. Cite the relevant PDF page number(s)."""
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Groq chat model",
        options=["openai/gpt-oss-120b", "llama-3.1-8b-instant"],
        index=0,
        help="If the first model is unavailable in your Groq account, try the second."
    )
    top_k = st.slider("Retrieved text chunks", min_value=2, max_value=8, value=5)
    st.caption("Embeddings: all-MiniLM-L6-v2 · Vector search: FAISS")

api_key = get_groq_key()
if not api_key:
    st.warning("Add GROQ_API_KEY in Streamlit Cloud → Settings → Secrets before asking questions.")
    st.code('GROQ_API_KEY = "your_groq_api_key_here"', language="toml")

uploaded_pdf = st.file_uploader("Upload a PDF timetable or document", type=["pdf"])

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        with st.spinner("Reading PDF and building the searchable knowledge base..."):
            try:
                import io
                pages = extract_pdf_pages(io.BytesIO(pdf_bytes))
                chunks = chunk_pages(pages)
                if not chunks:
                    st.error("No selectable text was found. This may be a scanned/image-only PDF. Run OCR on it or upload a text-based PDF.")
                    st.stop()
                embedding_model = load_embedding_model()
                index = build_faiss_index(chunks, embedding_model)
                st.session_state["file_hash"] = file_hash
                st.session_state["chunks"] = chunks
                st.session_state["faiss_index"] = index
                st.session_state["page_count"] = len(pages)
                st.session_state["chunk_count"] = len(chunks)
                st.session_state["chat_history"] = []
            except Exception as exc:
                st.error(f"Could not process this PDF: {exc}")
                st.stop()

    st.success(
        f"Document ready: {uploaded_pdf.name} · "
        f"{st.session_state.get('page_count', 0)} text pages · "
        f"{st.session_state.get('chunk_count', 0)} searchable chunks"
    )
    st.info("Example: “Which teacher teaches Corporate Finance on Monday, and what is the room number?”")

    question = st.chat_input("Ask a question about the uploaded PDF...")
    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not api_key:
                answer = "Please configure GROQ_API_KEY in Streamlit Cloud Secrets first."
            else:
                try:
                    with st.spinner("Searching the PDF and preparing an answer..."):
                        model = load_embedding_model()
                        retrieved = retrieve_chunks(
                            question,
                            st.session_state["chunks"],
                            st.session_state["faiss_index"],
                            model,
                            top_k=top_k,
                        )
                        answer = answer_question(question, retrieved, api_key, model_name)
                        with st.expander("Retrieved PDF evidence"):
                            for item in retrieved:
                                st.markdown(f"**PDF page {item['page']} · similarity {item['score']:.3f}**")
                                st.write(item["text"])
                except Exception as exc:
                    answer = f"An error occurred: {exc}"
            st.markdown(answer)
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
else:
    st.markdown("### How it works")
    st.markdown(
        "1. Upload a PDF.\n"
        "2. Extract text and split it into overlapping chunks.\n"
        "3. Convert chunks into sentence embeddings.\n"
        "4. Store embeddings in a FAISS vector index.\n"
        "5. Retrieve the most relevant chunks for your question.\n"
        "6. Send only those excerpts to Groq to generate a grounded answer."
    )
