import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from typing import Optional

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

import gradio as gr

# load_dotenv()

datapath_books_with_emotions = r'C:\\Users\\Lenovo\\Desktop\\LLM\\Semantic Book Recommender\\books_with_emotions.csv'
datapath_tagged_description = r'C:\\Users\\Lenovo\\Desktop\\LLM\\Semantic Book Recommender\\tagged_description.txt'

books = pd.read_csv(datapath_books_with_emotions)

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"


books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

#Method 1
# raw_documents = TextLoader("tagged_description.txt").load()
# text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)
# db_books = Chroma.from_documents(documents, OpenAIEmbeddings())


#Method 2
def split_by_lines(documents):
    """Split each document by newlines, creating one chunk per line."""
    result = []
    for doc in documents:
        lines = doc.page_content.split('\n')
        for line in lines:
            if line.strip():  # skip empty lines
                result.append(Document(page_content=line, metadata=doc.metadata))
    return result

raw_documents = TextLoader(datapath_tagged_description, encoding='utf-8').load()
documents = split_by_lines(raw_documents)


# emb = OpenAIEmbeddings(model="text-embedding-3-small",timeout=60)
# db_books = Chroma.from_documents(
#                                 documents,
#                                 embedding=emb,
#                                 persist_directory="books_db",
#                                 collection_metadata={"hnsw:space": "cosine"})
#Already created and saved the vector database, so loading it directly

emb = OpenAIEmbeddings(model="text-embedding-3-small",timeout=60)
db_books = Chroma(
    persist_directory="books_db",
    embedding_function=emb
)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard: # type: ignore
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()
    # dashboard.launch(server_name="0.0.0.0", server_port=7860)
    # dashboard.launch(share=True)
