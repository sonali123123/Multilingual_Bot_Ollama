from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from gtts import gTTS
import os
from uuid import uuid4
import langid
import ollama
import chromadb
from chromadb.config import Settings  # Import Settings correctly
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Correct import
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel
import torch
import chromadb.utils.embedding_functions as embedding_functions


app = FastAPI()

# Define directories
persist_directory = "./storage"
uploaded_pdfs_directory = "uploaded_pdfs"
os.makedirs(uploaded_pdfs_directory, exist_ok=True)
os.makedirs(persist_directory, exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(device)}")
# else:
#     device = torch.device("cpu")
#     print("CUDA is not available. Using CPU.")

ef = ef = embedding_functions.InstructorEmbeddingFunction(
model_name="hkunlp/instructor-xl", device="cuda")



# Define Pydantic model for the /ask endpoint
class AskRequest(BaseModel):
    query: str
    pdf_id: str
    lang: str = 'en-US'  # Optional: Default to English

def initialize_vectordb(pdf_path: str, filename: str):
    # Load PDF documents
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    client = chromadb.PersistentClient(path="./storage")
    collection_name = f"{filename}"
    collection = client.create_collection(name=collection_name,embedding_function=ef)
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)

    # Generate embeddings using Ollama and add to the collection with metadata
    for i, doc in enumerate(texts):
        try:
            #response = ollama.embeddings(model="mxbai-embed-large", prompt=doc.page_content)
            #embedding = response["embedding"]
            collection.add(
                ids=[f"{filename}_{i}"],  # Unique ID combining pdf_id and chunk index
                documents=[doc.page_content],
                metadatas=[doc.metadata]
            )
        except Exception as e:
            print(f"Error generating embeddings for document {i}: {e}")
            raise e
        

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    
    try:
        pdf_path = os.path.join(uploaded_pdfs_directory, file.filename)

        # Save the uploaded PDF to disk
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # Initialize the vector database with Ollama embeddings
        initialize_vectordb(pdf_path, file.filename)

        return {"message": "PDF uploaded and processed successfully."}
    except Exception as e:
        print(f"Error during PDF upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF: {e}")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index4.html", {"request": request})

@app.post("/ask")
async def ask(request: Request):
    
    try:
        print(Request)
        data = await request.json()
        print(data)
        
        user_input = data['query']
        print(user_input)
        filename = data['file']
        print(filename)
        #lang = request.lang

        #print(f"User Query: {user_input}, PDF ID: {filename}, Language: {lang}")

        collection_name = f"{filename}"

        if not user_input:
            return JSONResponse(content={"error": "No query provided."}, status_code=400)

        # Define filter to retrieve documents only from the specified PDF
        client = chromadb.PersistentClient(path="./storage")
        print(client)
        # ol_embed = ollama.embeddings(model="mxbai-embed-large")
        # embedding = ol_embed["embedding"]
        collection = client.get_collection(name=collection_name,embedding_function=ef)
        print(collection)
        #filter_criteria = {"pdf_id": {"$eq": pdf_id}}

        # Retrieve relevant documents with the filter
        resp_obj = collection.query(
            query_texts=user_input,
            n_results=3
        )
        print(resp_obj)

        # Extract documents from the response
        retrieved_docs = [doc for doc in resp_obj["documents"][0]]
        print(retrieved_docs)

        if not retrieved_docs:
            return JSONResponse(content={"error": "No relevant documents found for the given PDF ID."}, status_code=404)

        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question}
        """

        context_text = '\n\n---\n\n'.join(retrieved_docs)
        prompt = PROMPT_TEMPLATE.format(context=context_text, question=user_input)

        print("Generating response with Ollama")
        response = ollama.generate(
            model="llama3.1:8b",
            prompt=prompt
        )
        response_text = response['response']
        print(f"Generated Response: {response_text}")

        # Text-to-speech conversion
        input_lang, _ = langid.classify(user_input)
        tts = gTTS(response_text, lang=input_lang)
        audio_filename = f"{uuid4()}.mp3"
        audio_file_path = os.path.join("static", audio_filename)
        tts.save(audio_file_path)
        print(f"Audio saved to: {audio_filename}")

        return JSONResponse(content={"response": response_text, "audio_url": f"http://10.7.0.28:8000/static/{audio_filename}"})
    except Exception as e:
        print(f"Error during query processing: {str(e)}")
        return JSONResponse(content={"error": f"Failed to process query: {str(e)}"}, status_code=500)
    

# Optional: Endpoint to list all uploaded PDFs
@app.get("/list-col")
def list_col():
    client = chromadb.PersistentClient(path="./storage")
    collections = client.list_collections()
    collection_names = [collection.name for collection in collections]
    print(collection_names)
    return collection_names


    


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


    ##10.7.0.28:8000
                                                                                                                                                                                                                          
