from flask import Flask, jsonify, abort, request
import requests
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import io

os.environ.get('OPEN_AI_KEY')

app = Flask(__name__)

@app.route('/')
def process_pdf():
    try:
        # Get parameters from query string
        pdf_url = request.args.get('pdf_url')
        query = request.args.get('query')

        if not pdf_url or not query:
            # If pdf_url or query is missing, return a 400 Bad Request
            abort(400, "Both 'pdf_url' and 'query' parameters are required.")

        # Download the PDF content from the URL
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        reader = PdfReader(io.BytesIO(response.content))

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
        docs = docsearch.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)

        # Return the result as a JSON response
        return jsonify({'result': result})

    except requests.exceptions.RequestException as e:
        # Handle requests-related errors
        return jsonify({'error': f'Requests error: {str(e)}'}), 500

    except Exception as e:
        # Handle other unexpected errors
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run()

