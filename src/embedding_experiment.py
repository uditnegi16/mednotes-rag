# import time
# import pandas as pd
# import chromadb
# from sentence_transformers import SentenceTransformer

# # ==========================
# # Configuration
# # ==========================

# # Path to chunked dataset from Phase 3
# INPUT_PATH = "data/processed/chunks.csv"

# # Embedding model to test
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# # Chroma collection name
# COLLECTION_NAME = "embedding_experiment"

# # Number of chunks to embed for the experiment
# # (Keeps the experiment fast. We'll embed everything later.)
# SAMPLE_SIZE = 500


# def main():

#     print("=" * 60)
#     print("Embedding Experiment Started")
#     print("=" * 60)

#     # --------------------------
#     # Load chunk dataset
#     # --------------------------
#     df = pd.read_csv(INPUT_PATH)

#     # Use only first SAMPLE_SIZE chunks
#     df = df.head(SAMPLE_SIZE)

#     print(f"Loaded {len(df)} chunks")

#     # --------------------------
#     # Load embedding model
#     # --------------------------
#     print(f"\nLoading model: {MODEL_NAME}")

#     model = SentenceTransformer(MODEL_NAME)

#     # --------------------------
#     # Generate embeddings
#     # --------------------------
#     start = time.time()

#     embeddings = model.encode(
#         df["text"].tolist(),
#         show_progress_bar=True
#     )

#     embedding_time = time.time() - start

#     print(f"\nEmbedding completed in {embedding_time:.2f} seconds")

#     # --------------------------
#     # Create local Chroma database
#     # --------------------------
#     client = chromadb.Client()

#     # Delete old collection if it already exists
#     try:
#         client.delete_collection(COLLECTION_NAME)
#     except:
#         pass

#     collection = client.create_collection(COLLECTION_NAME)

#     # --------------------------
#     # Store vectors
#     # --------------------------
#     collection.add(
#         ids=df["chunk_id"].astype(str).tolist(),
#         documents=df["text"].tolist(),
#         embeddings=embeddings.tolist()
#     )

#     print(f"Stored {collection.count()} vectors")

#     print("\nExperiment completed successfully.")


# if __name__ == "__main__":
#     main()

import time
import json
import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer


# ==========================
# Configuration
# ==========================

# Chunk data created from Phase 3
INPUT_PATH = "data/processed/chunks.csv"

# Evaluation benchmark
EVAL_PATH = "evaluations/embedding_queries.json"

# Model being tested
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Temporary vector collection
COLLECTION_NAME = "embedding_eval"

# Number of results retrieved
TOP_K = 5

# Experiment size
SAMPLE_SIZE = 500



def load_questions():
    """
    Load evaluation questions.
    Each question has the expected relevant chunk id.
    """
    with open(EVAL_PATH, "r") as file:
        return json.load(file)



def calculate_mrr(results, expected_id):
    """
    Mean Reciprocal Rank.

    If correct chunk is:
    position 1 -> 1/1
    position 2 -> 1/2
    position 5 -> 1/5
    """

    for rank, chunk_id in enumerate(results, start=1):

        if chunk_id == expected_id:
            return 1 / rank

    return 0



def main():

    print("=" * 60)
    print("RAG Embedding Evaluation")
    print("=" * 60)


    # --------------------------
    # Load chunks
    # --------------------------

    df = pd.read_csv(INPUT_PATH)


    print(f"Chunks loaded: {len(df)}")



    # --------------------------
    # Load embedding model
    # --------------------------

    print(f"\nModel: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)



    # --------------------------
    # Create embeddings
    # --------------------------

    start = time.time()

    embeddings = model.encode(
        df["text"].tolist(),
        show_progress_bar=True
    )

    embedding_time = time.time() - start



    print(
        f"Embedding generation time: {embedding_time:.2f}s"
    )



    # --------------------------
    # Create vector DB
    # --------------------------

    client = chromadb.Client()


    try:
        client.delete_collection(COLLECTION_NAME)

    except:
        pass


    collection = client.create_collection(
        COLLECTION_NAME
    )



    # Store vectors
    collection.add(
        ids=df["chunk_id"].astype(str).tolist(),
        documents=df["text"].tolist(),
        embeddings=embeddings.tolist()
    )


    print(
        f"Stored vectors: {collection.count()}"
    )



    # --------------------------
    # Evaluation
    # --------------------------

    questions = load_questions()


    total_precision = 0
    total_recall = 0
    total_mrr = 0

    total_latency = 0


    print("\nRunning retrieval evaluation...\n")



    for item in questions:

        question = item["question"]

        relevant_keywords = [
            word.lower()
            for word in item["relevant_keywords"]
        ]


        # Convert query into embedding
        query_vector = model.encode(
            question
        )


        # Measure retrieval latency
        start = time.time()


        result = collection.query(
            query_embeddings=[
                query_vector.tolist()
            ],
            n_results=TOP_K
        )


        latency = time.time() - start

        total_latency += latency


        retrieved_ids = result["ids"][0]
        retrieved_docs = result["documents"][0]


        # --------------------------
        # Relevance checking
        # --------------------------

        relevant_count = 0


        for doc in retrieved_docs:

            doc_text = doc.lower()

            if any(
                keyword in doc_text
                for keyword in relevant_keywords
            ):
                relevant_count += 1



        # Precision@K
        precision = (
            relevant_count / TOP_K
        )


        # Recall@K
        # because we defined one expected answer group
        recall = (
            1 if relevant_count > 0 else 0
        )


        # MRR based on first relevant chunk
        mrr = 0

        for rank, doc in enumerate(retrieved_docs, start=1):

            if any(
                keyword in doc.lower()
                for keyword in relevant_keywords
            ):
                mrr = 1 / rank
                break



        total_precision += precision
        total_recall += recall
        total_mrr += mrr



        print(question)

        print("Retrieved IDs:")
        print(retrieved_ids)

        print(f"Relevant chunks: {relevant_count}")

        print(f"Precision@{TOP_K}: {precision:.2f}")
        print(f"Recall@{TOP_K}: {recall:.2f}")
        print(f"MRR: {mrr:.2f}")

        print("-"*50)


    count = len(questions)



    print("\n" + "="*60)

    print("FINAL RESULTS")

    print("="*60)


    print(
        f"Precision@{TOP_K}: "
        f"{total_precision/count:.3f}"
    )

    print(
        f"Recall@{TOP_K}: "
        f"{total_recall/count:.3f}"
    )

    print(
        f"MRR: "
        f"{total_mrr/count:.3f}"
    )

    print(
        f"Average latency: "
        f"{total_latency/count:.4f}s"
    )

    print(
        f"Embedding time: "
        f"{embedding_time:.2f}s"
    )



if __name__ == "__main__":
    main()