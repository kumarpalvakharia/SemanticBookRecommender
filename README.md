# Semantic Book Recommender

# Semantic Book Recommender

A semantic search engine for books powered by Large Language Models (LLMs). Unlike traditional keyword search, this application allows users to find books based on natural language descriptions, plot summaries, "vibes," or specific emotions (e.g., *"a dark mystery set in Victorian London"* or *"a book about a robot learning to love"*).

## 🚀 Features

* **Semantic Search:** Uses OpenAI embeddings and ChromaDB to find books based on meaning, not just exact keyword matches.
* **Emotion Filtering:** Filters books based on emotional tone (Happy, Suspenseful, Sad, etc.) extracted via LLM analysis.
* **Zero-Shot Classification:** Categorizes books into fiction/non-fiction using generative AI techniques.
* **Interactive Dashboard:** A clean, user-friendly web interface built with **Gradio**.

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **LLM Framework:** LangChain
* **Embeddings & Inference:** OpenAI API (`text-embedding-3-small` / `gpt-3.5-turbo`)
* **Vector Database:** ChromaDB
* **Interface:** Gradio
* **Data Processing:** Pandas, NumPy

## 📂 Project Pipeline

This project follows a specific data processing pipeline. The files are ordered by their execution flow:

### 1. Data Preparation (`data-exploration.ipynb`)
* **Purpose:** The first step in the pipeline. It loads the raw dataset, performs initial data cleaning (handling missing values), and explores the dataset structure.
* **Output:** A cleaned dataset ready for vectorization.

### 2. Search Mechanism (`vector-search.ipynb`)
* **Purpose:** This notebook establishes the core search logic. It demonstrates how to convert book descriptions into vector embeddings using OpenAI and store them in **ChromaDB** for similarity search.
* **Key Concept:** Proof-of-concept for the semantic search engine.

### 3. Classification (`text-classification.ipynb`)
* **Purpose:** Uses Zero-Shot Classification (via OpenAI) to categorize books into high-level genres (e.g., Fiction vs. Non-Fiction) where the original metadata might be missing or messy.

### 4. Sentiment Enrichment (`sentiment-analysis.ipynb`)
* **Purpose:** An advanced step that feeds book descriptions into an LLM to extract the dominant emotional tone (e.g., "Suspenseful," "Joyful," "Sad").
* **Outcome:** These emotions are saved as metadata to allow for "vibe-based" filtering in the final app.

### 5. Application (`gradio-dashboard.py`)
* **Purpose:** The final production file. It combines the cleaned data, the vector database, and the sentiment metadata into a polished web application.
* **Action:** Run this file to launch the search engine.

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
```

### 2. Set up a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
OPENAI_API_KEY=your_actual_api_key_here
```

### 5. Run the Application
```bash
python gradio-dashboard.py
```



