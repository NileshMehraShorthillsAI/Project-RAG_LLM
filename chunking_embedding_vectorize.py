'''Hugging Face BERT Embedding'''
import json
import faiss
import numpy as np
import os
from abc import ABC, abstractmethod
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
import torch

# Load Hugging Face Model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to Generate Embeddings

#Each vector is 384 dimensional vector according to the model
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.squeeze().numpy().tolist()

# Abstract Chunking Class
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str):
        pass

# Semantic Chunker (LangChain)
class SemanticChunker(BaseChunker):
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, text: str):
        return self.text_splitter.split_text(text)   #Returning list of strings

# FAISS Vector Database Class
class FAISSVectorDB:
    def __init__(self, index_file="faiss_index.bin", metadata_file="documents_metadata.json"):
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = None
        self.docs = []

        self.load_index()
        self.load_metadata()

    def add_documents(self, docs):
        embeddings = np.array([get_embedding(doc.page_content) for doc in docs], dtype=np.float32)

        #Shape of Embedding - (num_docs, 384)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])  

        faiss.normalize_L2(embeddings)   #Normalizing so that cosine similarity can be calculated just by Dot Product
        self.index.add(embeddings)
        self.docs.extend(docs)

        self.save_index()
        self.save_metadata()

    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_file)

    def load_index(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            print("✅ FAISS index loaded.")
        else:
            print("⚠️ No FAISS index found. A new one will be created.")

    def save_metadata(self):
        print(f"⚡ Saving {len(self.docs)} metadata entries to {self.metadata_file}...")  
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump([{"title": doc.metadata["title"], "content": doc.page_content} for doc in self.docs], f, indent=4)
        print("✅ Metadata file saved successfully!")


    def load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.docs = [Document(page_content=doc["content"], metadata={"title": doc["title"]}) for doc in json.load(f)]
            print("✅ Document metadata loaded.")
        else:
            print("⚠️ No metadata found. It will be created.")

# Function to Process JSON Data
def process_json(json_file, chunker, vector_db):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    docs = []
    for entry in data:
        title = entry["title"]
        content = entry["content"]
        chunks = chunker.chunk(content)
        '''Dividing the content into chunks and storing it into Document object with title and content is chunk which will be used while generating response'''
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={"title": title})
            docs.append(doc)

    print(f"✅ Adding {len(docs)} documents to FAISS")
    vector_db.add_documents(docs)

    print("⚡ Manually saving metadata...")
    vector_db.save_metadata()


if __name__ == "__main__":
    '''Initializing Chunker and vector'''
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
    vector_db = FAISSVectorDB(index_file="faiss_index.bin", metadata_file="documents_metadata.json")

    # try:
    #     test_embedding = get_embedding("Hello, test query")
    #     print("✅ Hugging Face BERT is working! Sample embedding:", test_embedding[:5])
    # except Exception as e:
    #     print("❌ API Connection Error:", e)

    if vector_db.index is None or vector_db.index.ntotal == 0:
        print("⚡ Processing JSON and building FAISS index...")
        process_json("wikipedia_history.json", chunker, vector_db)
        print("⚡ Manually saving metadata...")
        vector_db.save_metadata()
        print(f"\n✅ FAISS index built with {vector_db.index.ntotal} vectors.")
    else:
        print(f"\n✅ FAISS index already exists with {vector_db.index.ntotal} vectors.")
    vector_db.save_metadata()
    print(f"\n✅ FAISS index built with {vector_db.index.ntotal} vectors.")

print(f"Total vectors in FAISS index: {vector_db.index.ntotal}")
