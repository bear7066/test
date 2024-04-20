"""
Load PDF
"""
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader

loader = PyMuPDFReader()
documents = loader.load(file_path="./data/law.pdf")

"""
Import chinese model
"""
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
embed_model = HuggingFaceEmbedding(model_name="shibing624/text2vec-base-chinese")

# embed_model = OpenAIEmbedding(
#     model="text-embedding-3-small", embed_batch_size=256
# )

"""
Vectorization
"""
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1024)
index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter], embed_model=embed_model
)

"""
Input Question
"""
from llama_index.core import PromptTemplate
query_str = "介紹一些你有的法律知識"

query_prompt = (
    "你是一個法律助理，"
    "請介紹 {num_queries} 個法律："
    "Query: {query}\n"
    "Queries:\n"
)
query_gen_prompt = PromptTemplate(query_prompt)

def generate_queries(llm, query_str: str, num_queries: int = 3):
    fmt_prompt = query_prompt.format(
        num_queries = num_queries - 1, query = query_str
    )
    response = llm.complete(fmt_prompt)
    queries = response.text.split("\n")
    return queries
ans = generate_queries(llm, query_str, num_queries=3)
print(ans)

