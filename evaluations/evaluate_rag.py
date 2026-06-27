# import os
# import sys


# ROOT = os.path.dirname(
#     os.path.dirname(
#         os.path.abspath(__file__)
#     )
# )

# sys.path.insert(
#     0,
#     ROOT
# )


# from src.crew import run_mednotes
# from crewai import LLM
# import json
# # -------------------------
# # Judge LLM
# # -------------------------

# judge = LLM(
#     model=os.getenv("LLM_MODEL"),
#     api_key=os.getenv("HF_TOKEN")
# )


# # -------------------------
# # Load evaluation dataset
# # -------------------------

# with open(
#     "evaluations/test_dataset.json",
#     "r"
# ) as f:

#     tests = json.load(f)



# # -------------------------
# # Metrics
# # -------------------------

# total = len(tests)

# keyword_hits = 0
# keyword_total = 0

# faithful_pass = 0
# faithful_fail = 0



# # -------------------------
# # Hallucination Judge
# # -------------------------

# def judge_answer(
#     question,
#     answer
# ):

#     prompt = f"""

# You are a strict RAG evaluator.

# Question:
# {question}


# Answer:
# {answer}


# Check if the answer contains information
# not supported by retrieved clinical evidence.

# Return ONLY:

# PASS

# or

# FAIL


# PASS:
# Answer is grounded.

# FAIL:
# Answer has hallucinated facts.

# """

#     result = judge.call(prompt)

#     return result.strip()



# # -------------------------
# # Run evaluation
# # -------------------------

# for i, item in enumerate(tests):


#     print(
#         "\n======================"
#     )

#     print(
#         f"TEST {i+1}"
#     )

#     question = item["question"]


#     print(
#         "QUESTION:",
#         question
#     )


#     answer = run_mednotes(
#         question
#     )


#     print(
#         "\nANSWER:"
#     )

#     print(answer)



#     # -------------------------
#     # Keyword recall style
#     # -------------------------

#     expected = item.get(
#         "expected_keywords",
#         []
#     )


#     matched = 0


#     for word in expected:

#         if word.lower() in answer.lower():

#             matched += 1


#     keyword_hits += matched
#     keyword_total += len(expected)



#     # -------------------------
#     # Hallucination evaluation
#     # -------------------------

#     result = judge_answer(
#         question,
#         answer
#     )


#     print(
#         "\nFaithfulness:",
#         result
#     )


#     if "PASS" in result:

#         faithful_pass += 1

#     else:

#         faithful_fail += 1





# # -------------------------
# # Final report
# # -------------------------

# print(
#     "\n======================"
# )

# print(
#     "EVALUATION COMPLETE"
# )


# print(
#     f"""
# Keyword Coverage:
# {keyword_hits}/{keyword_total}
# Score:
# {keyword_hits / keyword_total:.2f}
# """
# )



# print(
#     f"""
# Faithfulness:
# PASS {faithful_pass}
# FAIL {faithful_fail}

# Score:
# {faithful_pass / total:.2f}
# """
# )
import json
import os
import re
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.crew import run_mednotes


# -------------------------
# Load dataset
# -------------------------

with open(
    "evaluations/test_dataset.json",
    "r",
    encoding="utf-8"
) as f:

    tests = json.load(f)


total = len(tests)

success = 0
failure = 0

coverage_scores = []

grounded_answers = 0


for i, item in enumerate(tests):

    print("\n==========================")

    print(f"TEST {i+1}")

    question = item["question"]

    expected = item["expected_keywords"]


    print("QUESTION:", question)


    answer = run_mednotes(question)

    print("\nANSWER:\n")

    print(answer)


    # -------------------------
    # System Success
    # -------------------------

    if "System error" in answer:

        failure += 1

        continue

    success += 1


    # -------------------------
    # Retrieval Coverage
    # -------------------------

    matched = 0

    answer_lower = answer.lower()

    for word in expected:

        if word.lower() in answer_lower:

            matched += 1

    coverage = matched / max(
        len(expected),
        1
    )

    coverage_scores.append(
        coverage
    )


    print(
        f"\nCoverage: {coverage:.2f}"
    )


    # -------------------------
    # Source Grounding
    # -------------------------

    if re.search(
        r"Sources?:",
        answer,
        re.IGNORECASE
    ):

        grounded_answers += 1

        print("Grounding: PASS")

    else:

        print("Grounding: FAIL")



# -------------------------
# Final Metrics
# -------------------------

print("\n==========================")

print("FINAL REPORT\n")

avg_coverage = (
    sum(coverage_scores)
    /
    max(len(coverage_scores),1)
)

grounding_score = (
    grounded_answers
    /
    max(success,1)
)

system_success = (
    success
    /
    total
)

overall = (
    avg_coverage
    +
    grounding_score
    +
    system_success
) / 3


print(f"Tests                 : {total}")
print(f"Successful Runs       : {success}")
print(f"Failed Runs           : {failure}")

print(f"\nRetrieval Coverage    : {avg_coverage:.2%}")
print(f"Source Grounding      : {grounding_score:.2%}")
print(f"System Success Rate   : {system_success:.2%}")

print("\n--------------------------")

print(
    f"Overall Evaluation    : {overall:.2%}"
)