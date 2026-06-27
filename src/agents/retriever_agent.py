# agents/retriever_agent.py
from crewai import Agent, LLM
from src.tools.chromadb_tool import (
    ChromaRetrieverTool
)
from dotenv import load_dotenv
import os

load_dotenv()
llm = LLM(
model=os.getenv("LLM_MODEL"),
api_key=os.getenv("HF_TOKEN")
)

# Create retrieval tool
retriever_tool = ChromaRetrieverTool()



retriever_agent = Agent(
    # Agent identity
    role="Clinical Information Retriever",


    # What it should achieve
    goal=(
        "Find the most relevant clinical "
        "information from medical notes."
    ),


    # Agent behavior context
    backstory=(
        "You are a medical document retrieval "
        "specialist. You search clinical notes "
        "and provide only relevant evidence."
    ),


    # Tools available
    tools=[
        retriever_tool
    ],
    llm=llm,

    # Prevent unnecessary actions
    verbose=False
)