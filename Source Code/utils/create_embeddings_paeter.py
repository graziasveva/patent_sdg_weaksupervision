"""
Script for creating embedding matrices from the results of llm_extraction.py (papers and patents JSONL files) using the 'mpi-inno-comp/paeter' model from sentence-transformers.

USAGE & REQUIREMENTS:
---------------------
- Requires the output JSONL files from llm_extraction.py (extracted_papers.jsonl, extracted_patents.jsonl).
- Requires the sentence-transformers library and the 'mpi-inno-comp/paeter' model.
- Required Python packages (install with pip):
    numpy
    srsly
    tqdm
    sentence-transformers
    argparse

Example usage:
    python create_embeddings_paeter.py \
        --papers extracted_papers.jsonl \
        --patents extracted_patents.jsonl \
        --output_dir ./embeddings_paeter \
        --device cuda

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
from sentence_transformers import SentenceTransformer

def embed_fields(documents, field_names, output_prefix, model):
    """
    Embed specified fields from a list of documents using a SentenceTransformer model (no batching).
    Saves each field's embedding matrix as a .npy file with the given prefix.
    """
    for field in field_names:
        texts = [doc.get(field, "") for doc in documents]
        output_file = f"{output_prefix}_{field}.npy"
        print(f"Encoding all texts for field: {field}")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        np.save(output_file, embeddings)

def main():
    parser = argparse.ArgumentParser(description="Create embedding matrices from LLM extraction results using Paeter model.")
    parser.add_argument('--papers', type=str, required=True, help='Path to extracted_papers.jsonl from llm_extraction.py')
    parser.add_argument('--patents', type=str, required=True, help='Path to extracted_patents.jsonl from llm_extraction.py')
    parser.add_argument('--output_dir', type=str, default='./embeddings_paeter', help='Directory to save the output .npy files')
    parser.add_argument('--device', type=str, default='cpu', help='Device for inference (cpu or cuda)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)

    # Load model
    model_name = 'mpi-inno-comp/paeter'
    model = SentenceTransformer(model_name, device=args.device)

    # Load documents
    papers = [doc for doc in srsly.read_jsonl(args.papers) if all(doc.get(f, "").strip() for f in ["function", "solution", "application", "main_text"])]
    patents = [doc for doc in srsly.read_jsonl(args.patents) if all(doc.get(f, "").strip() for f in ["function", "solution", "application", "main_text"])]

    # Embed fields for patents
    embed_fields(
        patents,
        field_names=["function", "solution", "application", "main_text"],
        output_prefix="patent",
        model=model
    )

    # Embed fields for papers
    embed_fields(
        papers,
        field_names=["function", "solution", "application", "main_text"],
        output_prefix="paper",
        model=model
    )

    print(f"Embeddings saved in {args.output_dir}")

if __name__ == "__main__":
    main()