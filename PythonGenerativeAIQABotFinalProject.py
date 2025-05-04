# QA Bot Web App Project: LangChain + LLM + Gradio

# ✅ Step 1: Install Required Libraries
!pip install -U langchain langchain-community langchain-huggingface faiss-cpu gradio pypdf pandas beautifulsoup4

# ✅ Step 2: Upload PDF
from google.colab import files
uploaded = files.upload()

# ✅ Step 3: Load Document
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(list(uploaded.keys())[0])
documents = loader.load()

# ✅ Step 4: Split Text
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# ✅ Step 5: Embed Chunks
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# ✅ Step 6: Setup QA Chain
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

HUGGINGFACEHUB_API_TOKEN = "your_token_here"  # Replace with your Hugging Face token
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    temperature=0.2,
    max_length=256,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ✅ Step 7: Gradio Interface
import gradio as gr

def ask_question(query):
    try:
        return qa_chain.run(query)
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(fn=ask_question, inputs="text", outputs="text", title="LangChain QA Bot")
iface.launch(share=True)
