"""
ALTERNATIVE EMBEDDINGS CREATION.

Script for creating embedding matrices from the results of llm_extraction.py (papers and patents JSONL files).
it's based on the Google GenAI embedding model: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings API
The reason is mainly because it's at today the provider with the 1st and 2nd best models in the MTEB arena leaderboard for English Retrieval and STS applications. 

We select the "text-embedding-005" model.

USAGE & REQUIREMENTS:
---------------------
- Requires the output JSONL files from llm_extraction.py (extracted_papers.jsonl, extracted_patents.jsonl).
- Requires Google GenAI credentials and access to the embedding model. see: https://cloud.google.com/docs/authentication/application-default-credentials
- Required Python packages (install with pip):
    numpy
    srsly
    tqdm
    google-genai
    argparse

Example usage:
    python create_embeddings.py \
        --papers extracted_papers.jsonl \
        --patents extracted_patents.jsonl \
        --output_dir ./embeddings \
        --batch_size 250 \
        --max_retries 3 \
        --configuration RETRIEVAL_DOCUMENT

About the configuration, it refers to https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types
RETRIEVAL_DOCUMENT:
    This configuration is used when you want to embed documents that will be used for retrieval.
RETRIEVAL_QUERY:
    This configuration is used when you want to embed queries that will be used for retrieval.
In our context, we use RETRIEVAL_QUERY for patents and RETRIEVAL_DOCUMENT for papers.

This script will:
- Load the JSONL files for papers and patents.
- Create embedding matrices for the fields: function, solution, application, and main_text for both papers and patents.
- Save the resulting numpy arrays in the output directory with clear names.
"""

import os
import argparse
import numpy as np
import srsly
from tqdm import tqdm
from google import genai
from google.genai.types import EmbedContentConfig
import time

# ----------------------
# Embedding Function
# ----------------------
def embed_fields(documents, field_names, output_prefix, client, batch_size=250, max_retries=3, retry_delay=5, configuration="RETRIEVAL_DOCUMENT"):
    """
    Embed specified fields from a list of documents using Google GenAI sequentially.
    Saves each field's embedding matrix as a .npy file with the given prefix.
    """
    def embed_and_save(text_list, output_file, field_name):
        all_embeddings = []
        total = len(text_list)
        for i in tqdm(range(0, total, batch_size), desc=f"Embedding {field_name}"):
            batch = text_list[i:i + batch_size]
            retries = max_retries
            while retries > 0:
                try:
                    response = client.models.embed_content(
                        model="text-embedding-005",
                        contents=batch,
                        config=EmbedContentConfig(task_type=configuration),
                    )
                    for emb in response.embeddings:
                        all_embeddings.append(np.array(emb.values))
                    break  # success
                except Exception as e:
                    print(f"[{field_name}] Error on batch {i // batch_size}: {e}")
                    retries -= 1
                    if retries == 0:
                        print(f"[{field_name}] FAILED after retries on batch {i // batch_size}")
                        raise
                    print(f"[{field_name}] Retrying batch {i // batch_size}... ({retries} retries left)")
                    time.sleep(retry_delay)
        embedding_matrix = np.stack(all_embeddings)
        np.save(output_file, embedding_matrix)

    for field in field_names:
        texts = [doc.get(field, "") for doc in documents]
        output_file = f"{output_prefix}_{field}.npy"
        print(f"Starting embedding for field: {field}")
        embed_and_save(texts, output_file, field)

# ----------------------
# Main CLI
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Create embedding matrices from LLM extraction results.")
    parser.add_argument('--papers', type=str, required=True, help='Path to extracted_papers.jsonl from llm_extraction.py')
    parser.add_argument('--patents', type=str, required=True, help='Path to extracted_patents.jsonl from llm_extraction.py')
    parser.add_argument('--output_dir', type=str, default='./embeddings', help='Directory to save the output .npy files')
    parser.add_argument('--batch_size', type=int, default=250, help='Batch size for embedding requests')
    parser.add_argument('--max_retries', type=int, default=3, help='Max retries for failed embedding requests')
    parser.add_argument('--retry_delay', type=int, default=5, help='Delay (seconds) between retries')
    parser.add_argument('--google_credentials', type=str, default=None, help='Path to Google credentials JSON (optional, overrides env var)')
    parser.add_argument('--project', type=str, required=True, help='Google Cloud project ID')
    parser.add_argument('--location', type=str, default='us-central1', help='Google Cloud location')
    args = parser.parse_args()

    # Set credentials if provided
    if args.google_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_credentials

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)

    # Initialize Google GenAI client
    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    # Load documents
    papers = [doc for doc in srsly.read_jsonl(args.papers) if all(doc.get(f, "").strip() for f in ["function", "solution", "application", "main_text"])]
    patents = [doc for doc in srsly.read_jsonl(args.patents) if all(doc.get(f, "").strip() for f in ["function", "solution", "application", "main_text"])]

    # Embed fields for patents
    embed_fields(
        patents,
        field_names=["function", "solution", "application", "main_text"],
        output_prefix="patent",
        client=client,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        configuration="RETRIEVAL_QUERY"
    )

    # Embed fields for papers
    embed_fields(
        papers,
        field_names=["function", "solution", "application", "main_text"],
        output_prefix="paper",
        client=client,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        configuration="RETRIEVAL_DOCUMENT"
    )

    print(f"Embeddings saved in {args.output_dir}")

if __name__ == "__main__":
    main()