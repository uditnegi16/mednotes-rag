import pandas as pd
import json
import random


# Input chunk file
INPUT_PATH = "data/processed/chunks.csv"

# Output evaluation file
OUTPUT_PATH = "evaluations/embedding_queries.json"


# Number of evaluation samples
SAMPLE_COUNT = 50



def main():

    # Load chunks
    df = pd.read_csv(INPUT_PATH)


    # Select random chunks
    samples = df.sample(
        SAMPLE_COUNT,
        random_state=42
    )


    evaluation_data = []


    for _, row in samples.iterrows():

        text = row["text"]


        # Create keywords from chunk text
        words = (
            text.replace(",", " ")
            .replace(".", " ")
            .split()
        )


        # keep useful words
        keywords = [
            w.lower()
            for w in words
            if len(w) > 5
        ]


        keywords = list(
            set(keywords[:5])
        )


        if not keywords:
            continue


        # Simple question template
        question = (
            "What information is mentioned "
            "in this clinical note?"
        )


        evaluation_data.append(
            {
                "question": question,
                "relevant_keywords": keywords
            }
        )


    with open(
        OUTPUT_PATH,
        "w"
    ) as f:

        json.dump(
            evaluation_data,
            f,
            indent=2
        )


    print(
        f"Created evaluation samples: {len(evaluation_data)}"
    )



if __name__ == "__main__":
    main()