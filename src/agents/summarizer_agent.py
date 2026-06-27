# agents/summarizer_agent.py
from crewai import Agent
from crewai import LLM
from dotenv import load_dotenv
import os

load_dotenv()
llm = LLM(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("HF_TOKEN")
)

summarizer_agent = Agent(

    # Agent identity
    role="Clinical Answer Summarizer",


    # Objective
    goal=(
    "Answer medical questions ONLY using "
    "retrieved evidence. "
    "Do not add outside information. "
    "Always include a short Sources section "
    "using the provided source metadata."
),


    # Behavior
    backstory=(
    "You summarize clinical evidence. "
    "Keep answers concise. "
    "At the end include sources exactly "
    "from retrieved evidence."
),
    llm=llm,
    verbose=False
)