import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

from tools import search_airline_policy, search_flights

# a simple page config for initial testing
st.set_page_config(page_title="AI Flight Agent", page_icon="✈️", layout="centered")
load_dotenv()

# Initialize Agent in Session State (Prevents reset on UI rerun)
if "agent" not in st.session_state:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash", 
        temperature=0.2,
        max_retries = 3,
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    tools = [search_airline_policy, search_flights]
    system_prompt = SystemMessage(
        content="You are a strict flight assistant. NEVER guess airports, cities, or dates. If missing, ask the user."
    )
    memory = MemorySaver()
    st.session_state.agent = create_react_agent(llm, tools, prompt=system_prompt, checkpointer=memory)
    st.session_state.config = {"configurable": {"thread_id": "streamlit_session_1"}}

# Initialize Chat History for the UI
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can find live flights or check airline rules. Where are you heading?"}
    ]

st.title("✈️ AI Flight & Rule Assistant")

# Render previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle New User Input
if prompt := st.chat_input("Ask about flights or baggage rules..."):
    # Display user's prompt on the screen
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Agent response
    with st.chat_message("assistant"):
        with st.spinner("Searching tools..."):
            inputs = {"messages": [HumanMessage(content=prompt)]}
            try:
                result = st.session_state.agent.invoke(inputs, config=st.session_state.config)
                final_message = result["messages"][-1].content
                
                # Display the response
                st.markdown(final_message)
                
                # Save to UI history
                st.session_state.messages.append({"role": "assistant", "content": final_message})
            except Exception as e:
                st.error(f"Error: {str(e)}")