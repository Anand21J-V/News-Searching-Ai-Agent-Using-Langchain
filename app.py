import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# Set API Keys
os.environ["GROQ_API_KEY"] = "x"
os.environ["TAVILY_API_KEY"] = "x"

# Streamlit UI
st.set_page_config(page_title="Live News Ai Agent Using Langchain", page_icon="🏏")
st.title("Live News Ai Agent Using Langchain")
st.markdown("Ask anything")

# Input box
query = st.text_input("Enter your question:", "Score of India vs England Match 2025")

# Button to trigger the agent
if st.button("Get Answer"):
    with st.spinner("Thinking... Please wait ⏳"):
        # Initialize LLM
        llm = ChatGroq(model="llama3-8b-8192")  # or llama-3.3-70b-versatile if available

        # Tool setup
        search_tool = TavilySearchResults()
        tools = [search_tool]

        # Load prompt and create agent
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        # Run agent
        try:
            response = agent_executor.invoke({"input": query})
            st.success("✅ Done!")
            st.subheader("🔎 Agent Response:")
            st.write(response["output"])
        except Exception as e:
            st.error(f"❌ Error: {e}")
