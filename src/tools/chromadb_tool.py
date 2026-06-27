import chromadb
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from rank_bm25 import BM25Okapi

from crewai.tools import BaseTool


from src.config import (
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    RETRIEVAL_K,
    RERANK_TOP_K
)



class ChromaRetrieverTool(BaseTool):


    name: str = "clinical_note_retriever"


    description: str = (
        "Searches clinical notes database "
        "using vector search, keyword search "
        "and reranking."
    )


    def _run(
        self,
        query: str,
        specialty: str = None
    ):


        if not query.strip():

            return "Empty query received."


        try:


            # ----------------------------
            # Load models
            # ----------------------------

            embedding_model = SentenceTransformer(
                EMBEDDING_MODEL
            )


            reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )


            # ----------------------------
            # Connect DB
            # ----------------------------

            client = chromadb.PersistentClient(
                path=VECTOR_DB_PATH
            )


            collection = client.get_collection(
                COLLECTION_NAME
            )



            # ----------------------------
            # Vector Search
            # ----------------------------

            query_vector = embedding_model.encode(
                query
            ).tolist()



            search_args = {

                "query_embeddings": [
                    query_vector
                ],

                "n_results": RETRIEVAL_K
            }



            if specialty:

                search_args["where"] = {
                    "specialty": specialty
                }



            vector_results = collection.query(
                **search_args
            )



            vector_docs = (
                vector_results["documents"][0]
            )


            vector_meta = (
                vector_results["metadatas"][0]
            )



            # ----------------------------
            # BM25 Keyword Search
            # ----------------------------

            all_data = collection.get(
                include=[
                    "documents",
                    "metadatas"
                ]
            )


            all_docs = (
                all_data["documents"]
            )


            all_meta = (
                all_data["metadatas"]
            )



            tokenized_docs = [
                doc.lower().split()
                for doc in all_docs
            ]



            bm25 = BM25Okapi(
                tokenized_docs
            )



            bm25_scores = bm25.get_scores(
                query.lower().split()
            )



            top_indices = np.argsort(
                bm25_scores
            )[-RETRIEVAL_K:]



            bm25_docs = [
                all_docs[i]
                for i in top_indices
            ]



            bm25_meta = [
                all_meta[i]
                for i in top_indices
            ]



            # ----------------------------
            # Merge results
            # ----------------------------

            combined = {}



            for doc, meta in zip(
                vector_docs,
                vector_meta
            ):

                combined[doc] = meta



            for doc, meta in zip(
                bm25_docs,
                bm25_meta
            ):

                combined[doc] = meta



            documents = list(
                combined.keys()
            )



            metadata = list(
                combined.values()
            )



            # ----------------------------
            # Reranking
            # ----------------------------

            pairs = [
                [
                    query,
                    doc
                ]
                for doc in documents
            ]



            if not documents:

                return {
                    "context": (
                        "No relevant clinical evidence "
                        "was found in the database."
                    ),
                    "sources": []
                }

            scores = reranker.predict(
                pairs
            )



            ranked = sorted(
                zip(
                    documents,
                    metadata,
                    scores
                ),
                key=lambda x: x[2],
                reverse=True
            )



            # ----------------------------
            # No relevant evidence found
            # ----------------------------

            if not final:

                return {
                    "context": (
                        "No relevant clinical evidence "
                        "was found in the database."
                    ),
                    "sources": []
                }



            # ----------------------------
            # Format answer
            # ----------------------------

            output = []

            for doc, meta, score in final:


                output.append(
                    f"""
SOURCE

Specialty:
{meta.get("specialty","unknown")}


CONTENT:

{doc}
"""
                )



            context_text = "\n\n".join(output)


            sources_text = "\n".join(
                [
                    f"- {meta.get('specialty','unknown')} | "
                    f"chunk: {meta.get('chunk_id','unknown')}"
                    for meta in metadata[:RERANK_TOP_K]
                ]
            )


            return f"""
            CLINICAL EVIDENCE:

            {context_text}


            SOURCES:

            {sources_text}
            """



        except Exception as e:


            return (
                f"Retrieval failed: {e}"
            )