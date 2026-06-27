import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer


# ----------------------------
# Configuration
# ----------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_FILE = "data/processed/chunks.csv"

VECTOR_DB_PATH = "data/vectorstore"

COLLECTION_NAME = "mednotes"



def main():

    print("=" * 60)
    print("Production Embedding Pipeline")
    print("=" * 60)


    # Load chunks
    df = pd.read_csv(CHUNK_FILE)

    print(
        f"Loaded chunks: {len(df)}"
    )


    # Load frozen embedding model
    model = SentenceTransformer(
        MODEL_NAME
    )


    # Create persistent Chroma DB
    client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH
    )


    # Create / replace collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )
    # Remove old data before rebuilding
    if collection.count() > 0:
        client.delete_collection(
            COLLECTION_NAME
        )

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        # ----------------------------
    # Prepare documents
    # ----------------------------

    documents = df["text"].tolist()


    ids = [
        str(x)
        for x in df["chunk_id"]
    ]


    metadatas = [
        {
            "specialty": row.get(
                "medical_specialty",
                "unknown"
            )
        }
        for _, row in df.iterrows()
    ]


    print("Generating embeddings in batches...")


    # Batch embedding generation
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True
    )


    # Convert numpy arrays to lists
    embeddings = embeddings.tolist()


    print("Saving vectors...")


    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )


    print(
        f"Stored vectors: {len(ids)}"
    )

    print("Vector DB ready")



if __name__ == "__main__":
    main()