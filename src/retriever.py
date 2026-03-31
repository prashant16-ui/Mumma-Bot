import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings


@st.cache_resource
def get_vectorstore():
    api_key = st.secrets["PINECONE_API_KEY"]

    pc = Pinecone(api_key=api_key)

    embeddings = download_hugging_face_embeddings()

    index_name = "maa-saathi"

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )

    return vectorstore