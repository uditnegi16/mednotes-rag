import re


def keyword_precision(
    answer,
    keywords
):

    found = 0

    for k in keywords:

        if re.search(
            k,
            answer,
            re.I
        ):
            found += 1


    return found / len(keywords)



def keyword_recall(
    answer,
    keywords
):

    return keyword_precision(
        answer,
        keywords
    )


def faithfulness_prompt(
    question,
    answer,
    context
):

    return f"""

Question:

{question}


Context:

{context}


Answer:

{answer}


Does the answer contain information
not supported by context?

Return:

PASS
or
FAIL

"""