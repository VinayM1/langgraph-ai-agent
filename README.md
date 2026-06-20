# 🤖 LangGraph AI Agent (RAG + Tools + Streamlit)

A production-ready AI agent built using **LangGraph**, **LangChain**, and **Streamlit**, capable of performing calculations, answering AI/ML questions using Retrieval-Augmented Generation (RAG), and dynamically invoking tools.

---

## 🚀 Features

* 🔗 **LangGraph Agent Workflow**
* 🧠 **RAG (Retrieval-Augmented Generation)** using FAISS
* 🛠️ **Tool Calling System**

  * Arithmetic tools (add, multiply, divide)
  * Knowledge retrieval (`search_docs`)
* ⚡ **Groq LLM Integration** (fast inference)
* 📚 **Custom Knowledge Base**
* 🌐 **Streamlit UI**
* 🔄 Auto FAISS index creation (on first run)
* 🧪 Debug + Tool execution view

---

## 🧠 Architecture

```
User Input
   ↓
LangGraph Agent
   ↓
LLM decides:
   → Call Tool
   → OR Respond Directly
   ↓
Tool Execution (if needed)
   ↓
Final Response
```

---

## 🛠️ Tech Stack

* **LangGraph** – Agent workflow orchestration
* **LangChain** – Tooling & abstractions
* **Groq** – LLM inference
* **Google Generative AI** – Embeddings
* **FAISS** – Vector database
* **Streamlit** – Frontend UI

---

## 📁 Project Structure

```
langgraph-ai-agent/
│
├── agent.py          # Core agent logic (LangGraph + tools)
├── app.py            # Streamlit UI
├── ingest.py         # Document ingestion + FAISS index
├── sample_docs/      # Knowledge base (text files)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup (Local)

### 1. Clone repo

```bash
git clone https://github.com/YOUR_USERNAME/langgraph-ai-agent.git
cd langgraph-ai-agent
```

---

### 2. Create environment

```bash
uv venv
.venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
uv add streamlit langchain langgraph langchain-groq langchain-google-genai faiss-cpu python-dotenv
```

---

### 4. Add API keys

Create `.env`:

```env
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
```

---

### 5. Build vector database

```bash
python ingest.py
```

---

### 6. Run app

```bash
streamlit run app.py
```

---

## 🌐 Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to **Streamlit Cloud**
3. Deploy from repo
4. Add secrets:

```toml
GROQ_API_KEY = "your_key"
GOOGLE_API_KEY = "your_key"
```

---

## 🧪 Example Queries

* "What is machine learning?"
* "Explain transformers"
* "Add 25 and 17"
* "Divide 10 by 0"
* "What is LangGraph?"

---

## 🧩 How It Works

* The **LLM decides** whether to answer directly or call a tool
* If tool is required → LangGraph routes execution
* `search_docs` retrieves context from FAISS
* Final response is generated using tool output + reasoning

---

## ⚠️ Notes

* FAISS index is **not stored in repo**
* It is **auto-generated on first run**
* Requires both:

  * Groq API key (LLM)
  * Google API key (embeddings)

---

## 🔥 Future Improvements

* Chat memory
* File upload for dynamic RAG
* Multi-agent system
* Better UI (ChatGPT-style)
* Logging & analytics

---

## 👨‍💻 Author

**Vinay Meena**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
