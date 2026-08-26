import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

# Import the tools we created in the previous step
from tools import search_airline_policy, search_flights

def main():
    # Load environment variables
    load_dotenv()
    
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("[!] Error: GEMINI_API_KEY or GOOGLE_API_KEY is missing from your .env file.")
        return
        
    if not os.getenv("SERPAPI_API_KEY"):
        print("[!] Error: SERPAPI_API_KEY is missing from your .env file.")
        return

    # Initialize the LLM
    print("[*] Initializing Gemini AI...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.7-flash", 
        temperature=0.2,
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )

    # Define the tools
    tools = [search_airline_policy, search_flights]

    # System Prompt
    system_prompt = SystemMessage(
        content="You are a strict flight assistant. NEVER guess airports, cities, or dates. If any of these are missing, ask the user to clarify before searching."
    )

    # Initialize Memory
    memory = MemorySaver()

    # Attach memory and prompt to the agent
    agent = create_react_agent(llm, tools, prompt=system_prompt, checkpointer=memory)

    print("[+] AI Flight Agent ready! Type 'exit' to quit.")
    print("-" * 50)
    
    # Define a thread ID so the agent groups these messages together
    config = {"configurable": {"thread_id": "flight_chat_1"}}

    # The Chat Loop
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down...")
                break
                
            if not user_input.strip():
                continue
            
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            print("\nThinking...")
            result = agent.invoke(inputs, config=config)
            
            final_message = result["messages"][-1]
            print(f"\nAgent: {final_message.content}")

        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"\n[!] An error occurred: {str(e)}")

if __name__ == "__main__":
    main()