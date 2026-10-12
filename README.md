# IBS PDF Assistant — RAG with Streamlit, Groq, Sentence Transformers, and FAISS

A Retrieval-Augmented Generation (RAG) web app for asking questions about uploaded PDFs such as an IBS Department timetable. It can answer questions about teachers, subjects, days, time slots, classes, sections, and rooms when the information is available in selectable PDF text.

## AI workflow

```text
User uploads PDF
      ↓
PDF text extraction (pypdf)
      ↓
Clean and split text into overlapping chunks
      ↓
Sentence Transformer embeddings (all-MiniLM-L6-v2)
      ↓
FAISS vector index
      ↓
User asks a question
      ↓
Embed question → retrieve top-k relevant chunks
      ↓
Send retrieved excerpts + question to Groq LLM
      ↓
Grounded answer with PDF page references
```

## Run locally (optional)

Python 3.10 or 3.11 is recommended.

1. Download or clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set your Groq API key as an environment variable, or configure Streamlit secrets.
5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy on GitHub + Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore`.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
4. In your deployed app, open **App settings → Secrets** and add:

   ```toml
   GROQ_API_KEY = "paste_your_groq_api_key_here"
   ```

5. Save/reboot the app, then upload a PDF and ask a question.

**Never commit your real API key to GitHub.** The `.streamlit/secrets.toml.example` file is only an example. Do not rename it to `secrets.toml` and commit a real key.

## Example questions

- Which teacher teaches Corporate Finance?
- What is the room number for the 10:00 AM class?
- Which subjects does Dr. Khan teach on Monday?
- What time does the IBS section's Investment Analysis class start?
- Is the teacher and room for this class mentioned in the PDF?

## Important limitations

- The PDF must contain selectable text. A scanned PDF made only of images needs OCR before this app can search it.
- Timetables with merged cells, tiny text, or complicated layouts may extract in the wrong reading order. Check the **Retrieved PDF evidence** panel if an answer looks incorrect.
- The app asks the model not to guess when the evidence is missing, but generated answers can still be wrong. Verify important timetable details against the cited PDF page.
- The vector index is built in memory for the current Streamlit session. It is not a permanent document database; the user uploads the PDF again in a new session.
- The first run downloads the Sentence Transformer model, so the first load may take some time.
- Large PDFs may take longer to embed and can use significant memory.

## Project files

- `app.py` — complete Streamlit app and RAG pipeline
- `requirements.txt` — Python dependencies
- `.streamlit/secrets.toml.example` — example API-key configuration
- `.gitignore` — excludes secrets and local Python files
