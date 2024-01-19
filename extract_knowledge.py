import argparse
import logging

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

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True)
args = vars(ap.parse_args())

pdf_reader = PdfReader(args['input'])
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

logging.info('Embedding texts...')
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)

logging.info('Starting...')
while True:
    question = input('You > ')
    docs = knowledge_base.similarity_search(question)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
    print(response)
