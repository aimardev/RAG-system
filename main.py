


import chromadb
from chromadb.utils import embedding_functions
import openai
import csv
import pandas as pd
import streamlit as st


CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)


collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func
)

openai.api_key = ""

context = """
You are AI assistant.
{}
"""

st.title("chat")
st.caption("Please ask what you want.")

question = st.text_input("Question")

if st.button("Get Answer"):
    query_result = collection.query(
        query_texts=[f"{question}"],
        n_results=1,
    )

    contents = ",".join(query_result["documents"][0])
    print("contents", contents)
    good_answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
         {"role": "system", "content": context.format(contents)},
         {"role": "user", "content": question},
     ],
    temperature=0,
    n=1,
    )

    st.write("Answer:", good_answer["choices"][0]["message"]["content"])
