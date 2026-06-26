import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Input file from Phase 2 (validated dataset)
INPUT_PATH = "data/processed/validated_data.csv"

# Output file for embeddings phase (next step)
OUTPUT_PATH = "data/processed/chunks.csv"

# Chunking hyperparameters
CHUNK_SIZE = 400      # number of words per chunk
OVERLAP = 60          # overlap words between chunks


def split_into_sentences(text):
    """
    Splits raw text into sentences using NLTK.
    (Not heavily used yet, but kept for future improvement)
    """
    return sent_tokenize(text)


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Converts long medical text into overlapping chunks.
    This is critical for retrieval quality in RAG systems.
    """

    # Split full text into words
    words = text.split()

    chunks = []

    # starting index of chunk window
    start = 0

    # loop until we consume all words
    while start < len(words):

        # end index of current chunk
        end = start + chunk_size

        # extract chunk words
        chunk_words = words[start:end]

        # join words back into text chunk
        chunk = " ".join(chunk_words).strip()

        # store non-empty chunk
        if len(chunk) > 0:
            chunks.append(chunk)

        # move window forward with overlap
        start = end - overlap

        # safety check to avoid negative index
        if start < 0:
            start = 0

    return chunks


def main():
    """
    Main pipeline:
    validated data → chunked dataset → saved for embeddings
    """

    # Load validated dataset from Phase 2
    df = pd.read_csv(INPUT_PATH)

    all_chunks = []

    print(f"Total documents: {len(df)}")

    # iterate over each medical record
    for idx, row in df.iterrows():

        text = str(row["text"])  # clinical note text
        doc_id = row["id"]       # unique document id
        specialty = row["medical_specialty"]  # medical category

        # generate chunks for each document
        chunks = create_chunks(text)

        # store each chunk with metadata
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc_id}_{i}",   # unique chunk identifier
                "doc_id": doc_id,              # parent document reference
                "text": chunk,                 # chunk content
                "medical_specialty": specialty,  # label metadata
                "chunk_index": i               # position inside document
            })

    # convert to dataframe
    chunk_df = pd.DataFrame(all_chunks)

    print(f"Total chunks created: {len(chunk_df)}")

    # save chunked dataset for embedding stage
    chunk_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()