# cfs25-sdg-concepts

## Mapping SDGs to patents through concept-extraction via LLMs
Author: Nicolò Tamagnone, Grazia Sveva Ascione

This project provides the main utilities for associating Sustainable Development Goals (SDGs) with patents and scientific papers using semantic embeddings, LLM-based extraction, weak supervision and hyperparameter optimization, as described in detail in the paper (sdg-concepts-paper.pdf). It was not possible to add the starting point data (SDG related scientific literature mainly), due to the size of the database. We provided anyway all the scripts used for builing the results described. 

---

## Pipeline overview

0. **Data download**
    - `download_paper.py`
    Utilities for over-limit query splitting and scientific literature download and saving in a mongo DB instance. In the Supporting Material and Tools, the SDG related queries and MongoDB docker-compose file are provided. 

1. **Data construction**
    - `final_dataset_construction.py`:  
      Merges SDG-labeled papers, patent-paper citations, and patent metadata to create a master dataset (`final_dataset.csv`) with all relevant fields for downstream processing.

2. **Concept extraction**
    - `llm_extraction.py`:  
      Uses an LLM (e.g. gpt-4.1-mini) to extract function, solution, and application concepts from the titles/abstracts of patents and papers.  
      Outputs: `extracted_patents.jsonl`, `extracted_papers.jsonl`.

3. **Embedding creation**
    - `create_embeddings_paeter.py`:  
      uses the `mpi-inno-comp/paeter` model from sentence-transformers for embedding creation.
    - `create_embeddings_google.py`:  
      Alternative: uses Google emebedding model to create semantic embeddings.


4. **Data alignment and splitting**
    - `data_alignment.py`:  
      Aligns the master dataset with the LLM-extracted JSONL files, aggregates SDG labels, and splits the data into training and validation sets using stratified multi-label splitting.  

5. **Semantic label generation**
    - `semantic_match.py`:  
      Uses the embeddings and SDG label matrix to propagate SDG labels from papers to patents via semantic similarity, using optimized thresholds and rank fusion.  
      Outputs:  
      - Training DataFrame with SDG label vectors (`sdg_vector`, `sdg_vector_norm`)

    - `semantic_match_optimaztion.py`:  
      Optimizes the semantic matching hyperparameters (thresholds, top-N) using validation data and Optuna.

6. **Supervised classifier training**
    - `classifier.py`:  
      Trains a transformer-based regression model (e.g., bert-for-patents) to predict SDG label distributions from patent main text, using the output of `semantic_match.py` as input.  
      Includes stratified multi-label splitting, class balancing, and evaluation.

---

For more details, see the docstrings at the top of each script.
If you need a master pipeline script or further automation, jusst ask. The project is composed of different steps, and it was complex to create a unique pipeline. 
