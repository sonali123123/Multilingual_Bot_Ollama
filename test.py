from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from gtts import gTTS
import os
from uuid import uuid4
import langid
from deep_translator import GoogleTranslator
import whisper
import shutil
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
import ollama
from langchain.chains import ConversationalRetrievalChain

app = FastAPI()

# Define directories
persist_directory = "./storage"
uploaded_pdfs_directory = "uploaded_pdfs"
os.makedirs(uploaded_pdfs_directory, exist_ok=True)
os.makedirs(persist_directory, exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mapping from pdf_id to retriever
pdf_retrievers = {}

def get_pdf_text(pdf_path: str):
    """Extract text from PDF using PyPDF2"""
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split the extracted PDF text into smaller chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def initialize_vectordb(pdf_path: str, pdf_id: str):
    """Initialize the vector database using FAISS"""
    global pdf_retrievers

    # Extract text from PDF
    raw_text = get_pdf_text(pdf_path)

    # Split text into chunks
    text_chunks = get_text_chunks(raw_text)

    # Generate embeddings and create FAISS vector store
    embeddings = OpenAIEmbeddings()  # Alternatively, you can use HuggingFaceInstructEmbeddings
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    # Store the retriever in the global dictionary
    pdf_retrievers[pdf_id] = vectorstore.as_retriever()

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_id = str(uuid4())
        pdf_filename = f"{pdf_id}_{file.filename}"
        pdf_path = os.path.join(uploaded_pdfs_directory, pdf_filename)

        # Save the uploaded PDF to disk
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # Initialize the vector database with FAISS embeddings
        initialize_vectordb(pdf_path, pdf_id)

        return {"message": "PDF uploaded and processed successfully.", "pdf_id": pdf_id}
    except Exception as e:
        print(f"Error during PDF upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF: {e}")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index4.html", {"request": request})

@app.post("/ask")
async def ask(request: Request):
    try:
        data = await request.json()
        print(f"Request Data: {data}")

        user_input = data.get('query')
        pdf_id = data.get('pdf_id')

        print(f"User Query: {user_input}, PDF ID: {pdf_id}")

        if not user_input:
            return JSONResponse(content={"error": "No query provided."}, status_code=400)

        if not pdf_id or pdf_id not in pdf_retrievers:
            return JSONResponse(content={"error": "Invalid or missing PDF ID."}, status_code=400)

        retriever = pdf_retrievers[pdf_id]
        print(f"Retrieving documents for query: {user_input}")
        resp_obj = retriever.get_relevant_documents(user_input)

        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question}
        """

        context_text = '\n\n---\n\n'.join([resp.page_content for resp in resp_obj])
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

        return JSONResponse(content={"response": response_text, "audio_url": f"/static/{audio_filename}"})
    except Exception as e:
        print(f"Error during query processing: {str(e)}")
        return JSONResponse(content={"error": f"Failed to process query: {str(e)}"}, status_code=500)

# Optional: Endpoint to list all uploaded PDFs
@app.get("/list-pdfs")
def list_pdfs():
    pdfs = []
    for pdf_id in pdf_retrievers.keys():
        pdfs.append({"pdf_id": pdf_id})
    return {"uploaded_pdfs": pdfs}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
