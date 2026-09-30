import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from google import genai
from dotenv import load_dotenv
from pathlib import Path


#load the env file which contains the api key for genai
load_dotenv()

client = genai.Client()


#loading the leagel file --> in our case it was a pdf
pdf_path = Path(__file__).parent/"2017-attention-is-all-you-need-Paper.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()


# chunking = split the doc into multiple chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap= 200  # context understanding
)
chunks = text_splitter.split_documents(docs)
print(f"Loaded {len(docs)} page(s), split into {len(chunks)} chunk(s).")


# converting the textual data into the vector embedding
embedding_model = GoogleGenerativeAIEmbeddings(
    model= "gemini-embedding-001"
)

# store the vector embedding into a vector database
vector_db = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",      # your Qdrant server
    collection_name="vector_embeddings",
)


print('indexing of the document done successfully')

vector_db.client.close()
