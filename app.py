import logging
import streamlit as st

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()
logging.getLogger().setLevel(logging.INFO)

st.header('Election LLM')

pdf = st.file_uploader('Upload the PDF of the candidate of your choice', type='pdf')

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    question = st.text_input('Ask about your candidate of choice')
    if question:
        docs = knowledge_base.similarity_search(question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type='stuff')
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=question)
        st.write(response)
