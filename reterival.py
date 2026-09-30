import os
# from wsgiref import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from google import genai
from dotenv import load_dotenv


#load the env file which contains the api key for genai
load_dotenv()

client = genai.Client()

# converting the textual data into the vector embedding
embedding_model = GoogleGenerativeAIEmbeddings(
    model= "gemini-embedding-001"
)


# retrive the relevent content from the vector databade 
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",      # your Qdrant server
    collection_name="vector_embeddings",
    embedding=embedding_model
)

# take the user Qurey as input
user_query = input("Enter your query: ")

#similarty search for the query
search_results= vector_db.similarity_search(
    query=user_query
)

reterival_content = "\n\n\n".join([
    f"Page Content: {result.page_content}\n"
    f"Page Number: {result.metadata.get('page_label', 'N/A')}\n"
    f"File Location: {result.metadata.get('source', 'N/A')}"
    for result in search_results
])


#system prompt to instruct the model to answer the query based on the retrieved content
SYSTEM_PROMPT = f"""
You are a helpful AI assistant who answers the user Query based on the avialable context
retrieved from the PDF files along with page content and page number.

You should only answeer the user based on the following context and navigate the
user to open the right page number to know more about it.

content={reterival_content}
"""

chat_response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=user_query,
    config={"system_instruction": SYSTEM_PROMPT},
)

print(f"chat_response: {chat_response.text}")