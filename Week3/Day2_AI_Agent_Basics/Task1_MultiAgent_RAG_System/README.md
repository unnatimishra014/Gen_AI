# 🧠 Multi‑Agent HR Helper (Streamlit, no API keys)

An interactive Streamlit app that demonstrates a tiny **multi‑agent RAG** system:
- **💼 Salary Agent** answers salary questions.
- **🛡️ Insurance Agent** answers insurance questions.
- **🧑‍⚖️ Coordinator** routes each query to the best agent using lightweight TF‑IDF similarity (pure Python, no external APIs).

## 🚀 Quick Start (Local)

> You only need Python 3.9+ and VS Code. No API keys required.

```bash
# 1) Create/activate a virtual environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Install Streamlit
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Open the printed local URL in your browser.

## ☁️ Deploy to Streamlit Community Cloud (free)

1. Push this folder to a **GitHub** repository.
2. Go to **share.streamlit.io** and pick *New app*.
3. Select your repo, set **Main file path** to `app.py`, and deploy.
4. (Optional) Add `data/` as a folder in the repo so the two text files are included.

No secrets or additional setup required.

## 📁 Project Structure

```
.
├─ app.py
├─ agents.py
├─ data/
│  ├─ salary.txt
│  └─ insurance.txt
├─ .streamlit/
│  └─ config.toml
├─ requirements.txt
└─ README.md
```

## ✅ Sample Queries
- *How do I calculate annual salary?*
- *What is included in my insurance policy?*

## 🧩 Notes
- This project intentionally avoids external LLMs/APIs and heavy ML dependencies to honor **zero‑key, minimal‑download** constraints.
- The retrieval is a small, self‑contained TF‑IDF in `agents.py` built from Python's standard library.
