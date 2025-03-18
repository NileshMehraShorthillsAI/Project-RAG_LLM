import os
import json
import faiss
import numpy as np
import streamlit as st
from google import genai
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

API_KEY = "YOUR-GEMINI-API-KEY"
client = genai.Client(api_key=API_KEY)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

log_file = "chat_log.json"

def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

index_file = "faiss_index.bin"
if not os.path.exists(index_file):
    raise FileNotFoundError(f"FAISS index file '{index_file}' not found.")
index = faiss.read_index(index_file)

metadata_file = "documents_metadata.json"
if not os.path.exists(metadata_file):
    raise FileNotFoundError(f"Metadata file '{metadata_file}' not found.")
with open(metadata_file, "r") as f:
    documents_metadata = json.load(f)

def process_query(query, top_k=5):
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(documents_metadata):
            doc = documents_metadata[idx]
            results.append(doc)
    return results

def generate_response(query):
    relevant_docs = process_query(query, top_k=5)
    if not relevant_docs:
        return "No relevant information found in the knowledge base."
    context = "\n".join([doc["content"][:500] for doc in relevant_docs])
    prompt = f"Based on the following information:\n{context}\n\nAnswer this question:\n{query}"
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    response_text = response.text.strip()
    log_interaction(query, response_text)
    return response_text

def log_interaction(user_query, assistant_response):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": user_query,
        "assistant_response": assistant_response
    }
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_entry)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)

def load_chat_history():
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return json.load(f)
    return []

st.set_page_config(page_title="RAG Chatbot", page_icon=":speech_balloon:")
st.title("RAG Chatbot")
st.caption("Developed by Nilesh Mehra")

# Sidebar for chat history
st.sidebar.title("Chat History")
chat_history = load_chat_history()
for chat in chat_history:
    with st.sidebar.expander(chat["timestamp"]):
        st.write(f"**User:** {chat['user_query']}")
        st.write(f"**Assistant:** {chat['assistant_response']}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

if "end_chat" not in st.session_state or not st.session_state.end_chat:
    user_query = st.chat_input("Enter your query:")
    if user_query:
        st.chat_message("user").markdown(user_query)
        response = generate_response(user_query)
        st.chat_message("assistant").markdown(response)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if st.button("End Chat"):
    st.session_state.end_chat = True
    st.write("Thank you for using the chatbot. Developed by Nilesh Mehra.")