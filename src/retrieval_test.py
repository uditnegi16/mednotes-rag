import chromadb
from sentence_transformers import SentenceTransformer


# ----------------------------
# Configuration
# ----------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

VECTOR_DB_PATH = "data/vectorstore"

COLLECTION_NAME = "mednotes"



def main():

    # Load embedding model
    model = SentenceTransformer(
        MODEL_NAME
    )


    # Connect to saved Chroma DB
    client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH
    )


    # Load existing collection
    collection = client.get_collection(
        COLLECTION_NAME
    )


    question = (
        "What medications was the patient taking?"
    )


    # Convert question to vector
    query_vector = model.encode(
        question
    ).tolist()


    # Search
    result = collection.query(
        query_embeddings=[
            query_vector
        ],
        n_results=3
    )


    print("=" * 60)
    print("Question:")
    print(question)

    print("\nRetrieved chunks:")


    for doc in result["documents"][0]:

        print("-" * 40)
        print(doc[:300])


if __name__ == "__main__":
    main()