# 🚀 NEXORA AI — Your Own Doubt-Solving AI Assistant

**Stop searching through hours of videos. Start finding answers instantly.**

NEXORA AI is an AI-powered doubt-solving assistant that helps students find answers from their learning resources using Retrieval-Augmented Generation (RAG).

## 💡 The Problem

Imagine watching a 2-hour YouTube lecture and learning several concepts. A few days later, you forget one specific topic.

Now you have to:
- Open the lecture again.
- Search through the video timeline.
- Skip multiple sections.
- Spend several minutes finding a short explanation.

Students often struggle to revisit specific concepts from previously watched lectures.

## 🎯 The Solution

NEXORA AI makes this process easier.

Simply ask your doubt in natural language. NEXORA searches through its knowledge base, retrieves relevant transcript sections, and generates a concise answer with the original video source and timestamp.

**Example:**

Question: "What is CSS Flexbox?"

NEXORA AI:
- Retrieves relevant transcript content.
- Generates a short explanation.
- Shows the relevant lecture.
- Provides a direct YouTube timestamp link.

## ✨ Features

- **AI Doubt Solving:** Ask questions in natural language.
- **Semantic Search:** Find relevant information based on meaning, not just keywords.
- **RAG Pipeline:** Generate answers using retrieved knowledge.
- **Timestamp Navigation:** Jump directly to the relevant point in a YouTube video.
- **Category Filters:** Narrow down searches by category and subcategory.
- **NSWH Filters:** Targeted knowledge retrieval.
- **Context-Aware Responses:** Answers are grounded in retrieved learning material.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Streamlit | User interface |
| Qdrant Cloud | Vector database |
| Sentence Transformers | Text embeddings |
| Groq API | LLM-powered responses |
| RAG | Retrieval-Augmented Generation |
| YouTube Transcripts | Knowledge source |

## 🏗️ Architecture

```text
YouTube Videos
      ↓
Transcript Extraction
      ↓
Text Chunking
      ↓
Sentence Transformers
      ↓
Vector Embeddings
      ↓
Qdrant Cloud
      ↓
User Question
      ↓
Semantic Search + Filters
      ↓
Relevant Context
      ↓
Groq LLM
      ↓
AI Answer + Video Timestamp
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

### 6. Run the application

Index your transcript knowledge base:

```bash
python Rag.py
```

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

## 📂 Project Structure

```text
NEXORA-AI/
│
├── transcripts/
│   └── *.json
│
├── Rag.py
├── app.py
├── requirements.txt
├── .env
└── README.md
```

## 🚧 Current Status

NEXORA AI is under active development.

- Initial transcript knowledge base implemented.
- Vector indexing and semantic retrieval implemented.
- RAG-based answer generation implemented.
- Streamlit interface developed.
- Knowledge base expansion and retrieval improvements ongoing.

## 🔮 Future Improvements

- Support for more learning resources.
- PDF and document-based doubt solving.
- Improved retrieval accuracy.
- Conversation history.
- Personalized knowledge collections.

## 🎯 Vision

**NEXORA AI aims to make learning resources easier to search, understand, and revisit.**

Less time searching. More time learning.

---

*Built with curiosity and a passion for AI.*
