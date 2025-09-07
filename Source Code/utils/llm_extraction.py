"""
Script for extracting Function, Solution, and Application concepts from patents and papers using an LLM (OpenAI via LangChain).
The LLM choice is based on our availability, and gpt-4.1-mini was the best choice in terms of speed, accuracy and cost. 
We leverage the function-tools capabilities of openai models, in order to have a more precise extraction.
The process can be easily extented to accept others LLM API or open source models.

USAGE & REQUIREMENTS:
---------------------
- This script uses only OpenAI models via LangChain (ChatOpenAI). It can be extended to other LLM providers with minimal changes.
- Required Python packages (install with pip):
    pandas
    langchain-openai
    langchain-core
    pydantic
    tqdm

- Input: CSV file with at least the following columns:
    - SDG (non-null rows will be processed)
    - oaid (for papers)
    - patent (for patents)
    - Title, Abstract (for papers)
    - appln_title, appln_abstract (for patents)

- Example usage:
    python llm_extraction.py --input all_sdg_doi_patent.csv --openai_key sk-... --output_dir ./results --max_workers 5

- Output: Two JSONL files in the output directory:
    - extracted_papers.jsonl
    - extracted_patents.jsonl

Each line in the output files is a JSONL object with the original row and the extracted fields.
"""
import os
import argparse
import pandas as pd
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
import concurrent.futures
import threading
import json
from tqdm import tqdm

# ----------------------
# Configuration & Prompt
# ----------------------

# This prompt instructs the LLM to extract the Function, Solution, and Application from a given technical text (paper or patent).
# It is used as the template for the structured extraction performed by the script.
USER_PROMPT = """
You are an expert analyst specializing in patents and scientific literature. 
Your task is to read technical texts and distill their core technological concepts into clear, concise sentences. 
You excel at identifying the main purpose of an invention or research (Function), explaining how it works (Solution), and stating where it can be applied (Application). 
Your summaries are precise, scientifically accurate, and easily understandable to researchers and professionals.

Text: 
{text}
"""

PROMPT = ChatPromptTemplate.from_template(USER_PROMPT)

class Extractor(BaseModel):
    function: str = Field(description="""Summarize the primary technological purpose, problem, or objective that the invention or research aims to address. They represent the intended outcome or goal, often articulated as "verb + object" constructions (e.g., "reduce emissions," "detect anomalies"). Functions express what is being achieved, not how it is accomplished.""",
                          title="Function",
                          default="")
    solution: str = Field(description="""Summarize the technical methods, mechanisms, or processes that enable the realization of the stated function. They specify how the function is achieved, detailing the specific techniques, devices, or procedures introduced by the invention or described in the paper.""",
                    title="Solution",
                    default="")
    application: str = Field(
        description="""Summarize the practical, industrial, or contextual settings where the solution can be implemented or has impact. They specify where or in what domain the invention or research findings are relevant (e.g., healthcare, agriculture, transportation).""",
        title="Application",
        default="")

save_lock = threading.Lock()

def save_result_to_jsonl(result, path):
    """Thread-safe append of a result dict to a JSONL file."""
    if not result:
        return
    with save_lock:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def process_row(row, chain, text_fields, path):
    """Process a single row (paper or patent), extract concepts, and save result."""
    output = {}
    try:
        
        text = ". ".join([str(row.get(f, "")) for f in text_fields if pd.notnull(row.get(f)) and str(row.get(f)).strip()])
        future_paper = chain.invoke({"text": text})
        try:
            extraction = future_paper.model_dump()
        except Exception as e:
            raise e
        else:
            output.update(paper_ex)
            if "Title" in text_fields:
                output["paper_id"] = int(row["oaid"])
            else:
                output["patent_id"] = row["patent"]

            output["main_text"] = text
            save_result_to_jsonl(output, path)

    except Exception as e:
        print(f"Error at row {row}: {e}")    
    return output

def extract_concepts(df, chain, text_fields, output_path, max_workers=5):
    """Extract concepts for all rows in a DataFrame and save to output_path."""
    rows = df.to_dict(orient="records")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda row: process_row(row, chain, text_fields, output_path), rows), total=len(rows)))

def main():

    parser = argparse.ArgumentParser(description="Extract Function, Solution, Application from patents and papers using LLM.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file (must contain SDG, oaid, patent, Title, Abstract, appln_title, appln_abstract columns)')
    parser.add_argument('--openai_key', type=str, default=None, help='OpenAI API key (overrides environment variable)')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output JSONL files')
    parser.add_argument('--max_workers', type=int, default=5, help='Number of parallel workers (default: 5)')
    args = parser.parse_args()

    # Set OpenAI API key
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    elif not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key must be provided via --openai_key or the OPENAI_API_KEY environment variable.")

    # Load data
    data = pd.read_csv(args.input)
    # we only process rows with SDG 
    data = data[data.SDG.notnull()]

    # Prepare dataframes
    papers = data.drop_duplicates("oaid").reset_index(drop=True)
    patents = data.drop_duplicates("patent").reset_index(drop=True)

    # Prepare output paths
    os.makedirs(args.output_dir, exist_ok=True)
    papers_out = os.path.join(args.output_dir, "extracted_papers.jsonl")
    patents_out = os.path.join(args.output_dir, "extracted_patents.jsonl")

    # Set up LLM chain
    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    # we use function calling to extract the structured output
    chain = PROMPT | model.with_structured_output(Extractor, method="function_calling")

    print(f"Processing {len(papers)} papers...")
    extract_concepts(papers, chain, ["Title", "Abstract"], papers_out, max_workers=args.max_workers)
    print(f"Saved paper results to {papers_out}")

    print(f"Processing {len(patents)} patents...")
    extract_concepts(patents, chain, ["appln_title", "appln_abstract"], patents_out, max_workers=args.max_workers)
    print(f"Saved patent results to {patents_out}")

if __name__ == "__main__":
    main()