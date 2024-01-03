# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_admin import initialize_app

# initialize_app()
#
#
# @https_fn.on_request()
# def on_request_example(req: https_fn.Request) -> https_fn.Response:
#     return https_fn.Response("Hello world!")


import requests
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

os.environ.get('OPEN_AI_KEY')
os.environ["OPENAI_API_KEY"] = "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"

# Replace 'your_pdf_url' with the actual URL of the PDF file
pdf_url = 'https://owll.massey.ac.nz/pdf/sample-book-review.pdf'

# Download the PDF from the URL
response = requests.get(pdf_url)
with open('/Users/muhammadmustafa/Desktop/downloaded_pdf.pdf', 'wb') as pdf_file:
    pdf_file.write(response.content)

# Read the PDF using PyPDF2
reader = PdfReader('downloaded_pdf.pdf')

# Read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# We need to split the text that we read into smaller chunks
text_splitter = CharacterTextSplitter(        
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Initialize OpenAI embeddings and FAISS vector store
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Example question and document similarity search
query = "who are the authors of the article?"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)

# You can continue with the rest of your code as before
