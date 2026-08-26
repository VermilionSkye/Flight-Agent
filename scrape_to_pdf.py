import os
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF

# The official policy URLs we want to ingest
URLS = {
    "air_india_baggage": "https://www.airindia.com/in/en/frequently-asked-questions/baggage.html",
    "indigo_baggage": "https://www.goindigo.in/baggage/baggage-allowance.html"
}

OUTPUT_DIR = "./data"

def clean_text(text):
    # FPDF's default font (Helvetica) is strict about Unicode. 
    # This strips out weird web characters (like emojis or smart quotes) to prevent crashes.
    return text.encode('latin-1', 'ignore').decode('latin-1')

def scrape_and_convert():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Airlines block bots, so we use a standard browser User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for name, url in URLS.items():
        print(f"[*] Scraping {name}...")
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"[!] Failed to fetch {name} (Status: {response.status_code})")
            continue
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Strip out navigation, footers, and scripts so the LLM doesn't read junk code
        for unwanted in soup(["script", "style", "nav", "footer", "header"]):
            unwanted.extract()
            
        # Extract the raw text and clean it
        raw_text = soup.body.get_text(separator='\n\n', strip=True)
        safe_text = clean_text(raw_text)

        print("[*] Converting text to PDF...")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        
        # multi_cell automatically handles line breaks and pagination
        pdf.multi_cell(0, 5, txt=safe_text)
        
        file_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
        pdf.output(file_path)
        print(f"[+] Successfully saved {file_path}")

if __name__ == "__main__":
    scrape_and_convert()