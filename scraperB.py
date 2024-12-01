import wikipedia
import wikipediaapi
import logging
from uuid import uuid4
import os

# Configurazione logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def initialize_wikipedia(lang='en', user_agent="WikipediaInfoGetter/1.0"):
    """
    Initializes the Wikipedia API with a custom user agent.
    
    Args:
        lang (str): Language code (default: 'en').
        user_agent (str): Custom user agent for the API.
    
    Returns:
        wikipediaapi.Wikipedia: Initialized Wikipedia API object.
    """
    wikipedia.set_lang(lang)  # Configura la lingua per il modulo wikipedia
    return wikipediaapi.Wikipedia(user_agent=user_agent, language=lang)

def search_and_fetch_pages(search_term, lang='en', max_pages=10):
    """
    Searches Wikipedia for a term and fetches the content of the first few pages.
    
    Args:
        search_term (str): The term to search on Wikipedia.
        lang (str): Language code (default: 'en').
        max_pages (int): Maximum number of pages to fetch (default: 10).
    
    Returns:
        list of tuples: List of tuples containing (page_title, page_content).
    """
    # Usa il modulo wikipedia per ottenere i risultati di ricerca
    search_results = wikipedia.search(search_term, results=max_pages)
    
    if not search_results:
        logging.warning(f"No results found for the search term: {search_term}")
        return []
    
    logging.info(f"Found {len(search_results)} results for '{search_term}'. Fetching up to {max_pages} pages...")

    # Usa wikipediaapi per scaricare i contenuti delle pagine
    wiki_wiki = initialize_wikipedia(lang=lang)
    page_data = []
    
    for title in search_results[:max_pages]:
        page = wiki_wiki.page(title)
        if page.exists():
            logging.info(f"Fetching page: {title}")
            page_data.append((title, page.text))
        else:
            logging.warning(f"Page '{title}' does not exist.")
    
    return page_data

def save_pages_to_file(pages, output_dir=".", lang='en'):
    """
    Saves page contents to individual text files.
    
    Args:
        pages (list of tuples): List of tuples (page_title, page_content).
        output_dir (str): Directory where files will be saved (default: current directory).
        lang (str): Language code to include in filenames.
    """
    # Crea la directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    for title, content in pages:
        sanitized_title = title.replace(" ", "_").replace("/", "_")  # Sanitize filenames
        file_name = f"{uuid4()}-{sanitized_title}-{lang}.txt"
        logging.info(f"Saving page '{title}' to {file_name}")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(content)

if __name__ == "__main__":
    # Termini di ricerca da cercare su Wikipedia
    search_terms = [
        "brigantaggio",
        "sfruttamento sud italia",
        "sfruttamento meridione",
        "lira",
        "banco di napoli",
        "banche"
    ]



    # Lingua
    language = "it"
    
    # Scarica le prime 10 pagine per ogni termine di ricerca
    for term in search_terms:
        pages = search_and_fetch_pages(term, lang=language, max_pages=3)
        if pages:
            save_pages_to_file(pages, output_dir=".", lang=language)
