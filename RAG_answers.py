import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from google import genai
from dotenv import load_dotenv
from tqdm import tqdm
import faiss
import time

# Initialize the Gemini API client
API_KEY = "YOUR-GEMINI-API-KEY"
client = genai.Client(api_key=API_KEY)

# Initialize Hugging Face BERT Model for embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings using Hugging Face BERT
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Load FAISS index
index_file = "faiss_index.bin"
if not os.path.exists(index_file):
    raise FileNotFoundError(f"FAISS index file '{index_file}' not found.")
index = faiss.read_index(index_file)

# Load document metadata
metadata_file = "documents_metadata.json"
if not os.path.exists(metadata_file):
    raise FileNotFoundError(f"Metadata file '{metadata_file}' not found.")
with open(metadata_file, "r") as f:
    documents_metadata = json.load(f)

# Function to process user query
def process_query(query, top_k=5):
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)
    faiss.normalize_L2(query_embedding)

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve relevant documents
    results = []
    for idx in indices[0]:
        if idx < len(documents_metadata):
            doc = documents_metadata[idx]
            results.append(doc)

    return results


# Function to generate response using Gemini API
def generate_response(query):
    try:
        relevant_docs = process_query(query, top_k=5)
        if not relevant_docs:
            return "No relevant information found in the knowledge base."

        # Prepare context for LLM
        context = "\n".join([doc["content"][:500] for doc in relevant_docs])

        # Prompt to the Gemini
        prompt = f"Based on the following information:\n{context}\n\nAnswer this question:\n{query}"

        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )

        return response.text.strip()
    except Exception as e:
        return "An error occurred while generating the response."

# Function to read questions from JSON file and generate answers
def generate_answers(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        questions = json.load(infile)

    qa_pairs = []
    for item in tqdm(questions, desc="Processing Questions"):
        question = item.get("question", "")
        if question:
            answer = generate_response(question)
            qa_pairs.append({"question": question, "answer": answer})
            # Save each QA pair to the output file
            with open(output_file, "a", encoding="utf-8") as outfile:
                json.dump({"question": question, "answer": answer}, outfile, ensure_ascii=False)
                outfile.write("\n")
            time.sleep(1.1)  # Sleep for 1.1 seconds to stay within the rate limit
    print(f"✅ Successfully generated answers for {len(qa_pairs)} questions!")

# Specify input and output files
input_file = "generated_ques_ans.json"
output_file = "RAG_generated_qa_pairs.json"

# Generate answers and save to output file
generate_answers(input_file, output_file)
