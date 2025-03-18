import requests
from bs4 import BeautifulSoup
import json
import time
import random
from tqdm import tqdm

WIKI_BASE_URL = "https://en.wikipedia.org"

# Function to extract valid Wikipedia links
def get_wiki_links(url, visited):
    """Extracts valid Wikipedia history-related links from a given page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith("/wiki/") and ":" not in href and href not in visited:
            full_url = WIKI_BASE_URL + href
            links.add(full_url)

    return links

# Function to scrape content from a Wikipedia page
def scrape_wikipedia_page(url):
    """Scrapes the title and main content of a Wikipedia page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract the title
    title = soup.find("h1").text.strip()
    
    # Extract the main content
    content = []
    for p in soup.find_all("p"):
        text = p.text.strip()
        if text:
            content.append(text)

    return {"title": title, "content": " ".join(content)}

# Starting from the Wikipedia page - history of India
start_url = "https://en.wikipedia.org/wiki/History_of_India"
visited_pages = set()
scraped_data = []

# Get initial links from the main History of India page
wiki_links = get_wiki_links(start_url, visited_pages)

print(f"✅ Found {len(wiki_links)} potential pages to scrape.")

# Scrape pages up to 1000
for url in tqdm(wiki_links, desc="Scraping Wikipedia", total=1000):
    if len(scraped_data) >= 1000:
        break

    if url not in visited_pages:
        try:
            page_data = scrape_wikipedia_page(url)
            scraped_data.append(page_data)
            visited_pages.add(url)
        except Exception as e:
            print(f" Error scraping {url}: {e}")
        
        # Random sleep to avoid getting blocked
        time.sleep(random.uniform(1, 3))

# Saving the scraped data as JSON
with open("wikipedia_history.json", "w", encoding="utf-8") as f:
    json.dump(scraped_data, f, indent=4, ensure_ascii=False)

print(f"✅ Successfully scraped {len(scraped_data)} Wikipedia pages!")
