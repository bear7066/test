from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

documents = SimpleDirectoryReader("data").load_data()

embed_model = HuggingFaceEmbedding(model_name="shibing624/text2vec-base-chinese")

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

query_engine = index.as_query_engine()
response = query_engine.query("請問誰是申請人")
print(response)