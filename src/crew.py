from crewai import Crew, Task, Process

from src.logger import logger

from src.agents.query_agent import query_agent
from src.agents.retriever_agent import retriever_agent
from src.agents.summarizer_agent import summarizer_agent


def run_mednotes(question: str):

    if not question.strip():

        logger.error("Empty user query")

        return "Please enter a valid question."


    logger.info(
        f"User query: {question}"
    )


    query_task = Task(

        description=(
            f"Rewrite this medical question "
            f"for retrieval: {question}"
        ),

        expected_output=(
            "Optimized medical search query"
        ),

        agent=query_agent
    )


    retrieval_task = Task(

        description=(
            f"Search the clinical database for "
            f"{question}. "
            "Return relevant clinical evidence."
        ),

        expected_output=(
            "Retrieved clinical evidence "
            "with source metadata."
        ),

        agent=retriever_agent
    )


    summary_task = Task(

        description=(
            f"Answer this question: {question}. "
            "Use only retrieved clinical evidence. "
            "Do not invent information."
        ),

        expected_output=(
            "Grounded medical answer."
        ),

        agent=summarizer_agent
    )


    crew = Crew(

        agents=[
            query_agent,
            retriever_agent,
            summarizer_agent
        ],

        tasks=[
            query_task,
            retrieval_task,
            summary_task
        ],

        process=Process.sequential,

        verbose=False
    )


    logger.info(
        "Crew execution started"
    )


    try:

        result = crew.kickoff()

        logger.info(
            "Crew execution completed"
        )

        answer = result.raw.strip()

        if (
            "No relevant clinical evidence" in answer
            or
            "not found in the database" in answer
        ):

            return (
                "The requested information was not "
                "found in the retrieved clinical notes."
            )

        return answer


    except Exception as e:

        logger.exception(e)

        return f"System error: {str(e)}"


if __name__ == "__main__":

    question = input(
        "\nEnter your medical question: "
    )

    answer = run_mednotes(question)

    print("\nFINAL ANSWER\n")

    print(answer)