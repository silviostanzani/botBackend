import os
from fastapi import FastAPI, Query
'''
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
'''
'''
# --- Configure LLM + embeddings for Azure OpenAI ---
# Create these App Service settings (Configuration > Application settings)
# AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION,
# AZURE_OPENAI_DEPLOYMENT (for the chat/completions model),
# AZURE_OPENAI_EMBEDDING_DEPLOYMENT (for text-embedding model)

Settings.llm = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ.get("OPENAI_API_VERSION", "2024-08-01-preview"),
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],        # deployment name of your gpt-* model
)
Settings.embed_model = AzureOpenAIEmbedding(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ.get("OPENAI_API_VERSION", "2024-08-01-preview"),
    model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]  # deployment name of your embedding model
)

STORAGE_DIR = "./storage"

def get_or_build_index():
    if os.path.isdir(STORAGE_DIR) and os.listdir(STORAGE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)
    # First run: read files under ./data and build the vector index
    docs = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index

index = get_or_build_index()
query_engine = index.as_query_engine(response_mode="compact")
'''

app = FastAPI(title="LlamaIndex on Azure App Service")

#class QueryIn(BaseModel):
#    q: str

@app.get("/")
def root():
    print('ok')
    return {"status": "ok", "message": "LlamaIndex + FastAPI on Azure"}

@app.post("/query")
def query():#body: QueryIn):
    answer = 'test'
    #answer = query_engine.query(body.q)
    print('answer')
    print(answer)
    return {"answer": str(answer)}


