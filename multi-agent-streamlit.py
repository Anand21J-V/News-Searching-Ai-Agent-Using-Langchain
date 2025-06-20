import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Set API Keys
os.environ["GROQ_API_KEY"] = "x"
os.environ["TAVILY_API_KEY"] = "x"

# Streamlit UI Setup
st.set_page_config(page_title="🧠 Multi-Agent Topic Analyzer", page_icon="🧠")
st.title("🧠 Multi-Agent Topic Analyzer")
st.markdown("Enter any trending or general topic and get a summary + critical analysis using AI agents.")

# User Input
query = st.text_input("🔍 Enter a topic:", "India's space mission")

if st.button("Analyze"):
    with st.spinner("⚙️ Agents are thinking..."):

        # Load LLM and tools
        llm = ChatGroq(model="llama3-8b-8192")
        search_tool = TavilySearchResults()

        # Agent 1 - Web Search & Summarizer
        react_prompt = hub.pull("hwchase17/react")
        agent1 = create_react_agent(llm=llm, tools=[search_tool], prompt=react_prompt)
        agent_executor1 = AgentExecutor(agent=agent1, tools=[search_tool], verbose=True, handle_parsing_errors=True)

        # Agent 2 - Critical Analysis Tool
        @tool
        def analyze_critically(text: str) -> str:
            """Provides critical analysis of the given information."""
            prompt = f"Critically analyze the following news summary:\n\n{text}\n\nDiscuss possible implications, concerns, or biases."
            return llm.invoke(prompt)

        agent2 = create_react_agent(llm=llm, tools=[analyze_critically], prompt=react_prompt)
        agent_executor2 = AgentExecutor(agent=agent2, tools=[analyze_critically], verbose=True, handle_parsing_errors=True)

        # Run Agent 1
        st.subheader("🔍 Agent 1: Summary from Web")
        summary_output = agent_executor1.invoke({"input": f"Summarize: {query}"})
        st.success(summary_output["output"])

        # Run Agent 2
        st.subheader("🧠 Agent 2: Critical Analysis")
        analysis_output = agent_executor2.invoke({"input": summary_output["output"]})
        st.info(analysis_output["output"])
