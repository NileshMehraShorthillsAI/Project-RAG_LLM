import requests
import json
import random
import time
from tqdm import tqdm

API_KEY = "YOUR-DEEPSEEK-API-KEY"
API_URL = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def generate_qa(title, paragraph):
    prompt = f"""You are an AI assistant that generates high-quality question-answer pairs for evaluating a Retrieval-Augmented Generation (RAG) model.

Given the following context:

**Title:** {title}
**Paragraph:** {paragraph}

Generate a well-formed question that can be answered from the paragraph and provide a precise answer.

Format your response as:
{{
  "question": "Generated question?",
  "answer": "The correct answer extracted from the paragraph."
}}
"""

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()  # Will raise an error for HTTP errors
        result = response.json()
        
        #For Debugging
        print(f"Raw API Response: {result}")

        output_text = result.get('choices', [{}])[0].get('message', {}).get('content', "").strip()

        if not output_text:
            print("⚠️ API returned empty content")
            return None
        
        qa_pair = json.loads(output_text)
        return qa_pair

    except requests.exceptions.RequestException as e:
        print(f"🚨 API Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"🚨 JSON decoding failed: {e}\nResponse content: {response.text}")
        return None

with open("wikipedia_history.json", "r", encoding="utf-8") as infile:
    try:
        data = json.load(infile)  
    except json.JSONDecodeError as e:
        print(f"❌ JSON decoding failed: {e}")
        exit()

# Opening  output file for writing
with open("generated_questions_deepseek.json", "w", encoding="utf-8") as outfile:
    # Initialize a list to collect QA pairs
    qa_pairs = []

    # Iterate over each entry in the JSON file
    for entry in tqdm(data, desc="Processing Wikipedia Entries"):
        title = entry.get("title", "Unknown Title")
        content = entry.get("content", "")

        # Select a meaningful paragraph (with more than 50 words)
        paragraphs = [p for p in content.split(". ") if len(p.split()) > 50]
        if not paragraphs:
            continue

        selected_paragraph = random.choice(paragraphs)
        qa_pair = generate_qa(title, selected_paragraph)
        
        if qa_pair:
            qa_pairs.append(qa_pair)

            outfile.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")

        # Sleep a random interval to avoid hitting rate limits
        time.sleep(random.uniform(1, 3))

print(f"✅ Successfully generated {len(qa_pairs)} QA pairs using DeepSeek!")


