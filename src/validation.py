import pandas as pd
import re
import hashlib

INPUT_PATH = "data/raw/mtsamples/mtsamples.csv"
OUTPUT_PATH = "data/processed/validated_data.csv"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # remove weird whitespace
    text = re.sub(r"\s+", " ", text)

    # remove excessive special characters (noise filter)
    special_char_ratio = len(re.findall(r"[^a-zA-Z0-9., ]", text)) / max(len(text), 1)
    if special_char_ratio > 0.25:
        return ""

    return text.strip()


def generate_id(text):
    return hashlib.md5(text.encode()).hexdigest()


def is_valid(text):
    if not text:
        return False
    if len(text) < 200:
        return False
    return True


def main():
    df = pd.read_csv(INPUT_PATH)

    print(f"Raw rows: {len(df)}")

    # keep only transcription column
    df = df[["transcription", "medical_specialty"]]

    # clean text
    df["clean_text"] = df["transcription"].apply(clean_text)

    # filter invalid
    df = df[df["clean_text"].apply(is_valid)]

    # drop duplicates
    df = df.drop_duplicates(subset=["clean_text"])

    # create structured dataset
    df_final = pd.DataFrame({
        "id": df["clean_text"].apply(generate_id),
        "text": df["clean_text"],
        "medical_specialty": df["medical_specialty"],
        "source": "mtsamples"
    })

    print(f"Validated rows: {len(df_final)}")

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()