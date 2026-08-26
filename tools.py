import os
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()

# --- 1. Initialize RAG Components ---
PERSIST_DIR = "./vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the local vector database we built in the last step
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="airline_policies"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 2. Define the Agent Tools ---

@tool
def search_airline_policy(query: str) -> str:
    """
    Use this tool to look up airline rules, baggage allowances, cancellation policies,
    or pet travel guidelines.
    Args:
        query: A specific search query (e.g., 'What is the Air India cabin baggage allowance?')
    """
    print(f"\n[Tool] Searching local DB for: {query}")
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant policy information found in the database."
    
    # Combine the top 3 chunks into a single text block for the LLM to read
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


@tool
def search_flights(departure_id: str, arrival_id: str, outbound_date: str, return_date: str = None) -> str:
    """
    Use this tool to find live flight prices, schedules, and durations.
    Args:
        departure_id: The 3-letter IATA airport code for departure (e.g., 'CDG', 'DEL', 'JFK').
        arrival_id: The 3-letter IATA airport code for arrival (e.g., 'LAX', 'LHR').
        outbound_date: The departure date in YYYY-MM-DD format.
        return_date: (Optional) The return date in YYYY-MM-DD format for round trips.
    """
    print(f"\n[Tool] Fetching live flights from {departure_id} to {arrival_id}...")
    
    params = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "currency": "USD",
        "hl": "en",
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    
    if return_date:
        params["return_date"] = return_date
        
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        best_flights = results.get("best_flights", [])
        
        if not best_flights:
            return "No flights found for this route and date combination."
            
        # Format the top 3 flights into a clean string for the LLM
        output = "Here are the top flight options:\n"
        for i, flight in enumerate(best_flights[:3]):
            flights_info = flight.get("flights", [{}])[0]
            airline = flights_info.get("airline", "Unknown Airline")
            price = flight.get("price", "Unknown Price")
            departure_time = flights_info.get("departure_airport", {}).get("time", "Unknown Time")
            arrival_time = flights_info.get("arrival_airport", {}).get("time", "Unknown Time")
            
            output += f"{i+1}. {airline} - ${price} | Departs: {departure_time}, Arrives: {arrival_time}\n"
            
        return output
        
    except Exception as e:
        return f"Error fetching flights: {str(e)}"