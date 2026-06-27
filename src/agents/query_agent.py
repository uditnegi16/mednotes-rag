from crewai import Agent, LLM
from dotenv import load_dotenv
import os


load_dotenv()


llm = LLM(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("HF_TOKEN")
)



query_agent = Agent(

    role="Medical Query Optimizer",

    goal=(
        "Rewrite user questions into precise "
        "medical retrieval queries."
    ),

    backstory=(
        "You improve medical search queries. "
        "Convert vague questions into "
        "clinical keywords without answering."
    ),

    llm=llm,

    verbose=False
)