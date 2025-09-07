"""
Semantic label generation for SDG-patent-paper association (training set).

This script provides a function to generate SDG label vectors for the training data using the best hyperparameters found in semantic_match_optimaztion.py.

Inputs:
- Patent embedding matrices (functions, solutions, applications) as .npy files (order must match the dataset rows).
- Paper embedding matrices (functions, solutions, applications) as .npy files.
- paper_sdg_labels: Paper-to-SDG association matrix (numpy array, shape: num_papers x num_sdgs).
- data_train: DataFrame with metadata for the training patents, must be aligned with the patent embeddings (order and size), and contain at least 'patent_id' and 'patent_text'.
- Best hyperparameters (thresholds, top_n) from optimization (set in the script below).

Outputs:
- For each patent, generates a vector of SDG label counts (and normalized version) based on semantic similarity to papers.
- Appends these vectors to the DataFrame as 'sdg_vector' and 'sdg_vector_norm', and saves as a CSV.

Usage example:
    from semantic_match import generate_sdg_vectors, get_aligned_train_val_split
    generate_sdg_vectors(
        dataset_path='aligned_patent_metadata.csv',
        functions_patent_path='patent_function.npy',
        solutions_patent_path='patent_solution.npy',
        applications_patent_path='patent_application.npy',
        functions_paper_path='paper_function.npy',
        solutions_paper_path='paper_solution.npy',
        applications_paper_path='paper_application.npy',
        paper_sdg_labels_path='paper_sdg_labels.npy',
        output_path='output_with_sdg_vectors.csv',
        batch_size=5000
    )
    train_df, val_df = get_aligned_train_val_split(aligned_df)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from skmultilearn.model_selection import iterative_train_test_split


dataset_path = ''  # Path to aligned patent metadata CSV
functions_patent_path = ''
solutions_patent_path = ''
applications_patent_path = ''
functions_paper_path = ''
solutions_paper_path = ''
applications_paper_path = ''
paper_sdg_labels_path = ''
output_path = ''

# our best tuned hyperparameters
sem_threshold_func = 0.2604440815587794
sem_threshold_sol = 0.16463215087216873
sem_threshold_app = 0.8727727292417202
top_n = 30
batch_size = 5000

# ---- Functions (same as tuning) ----
def compute_similarity_matrix(patent_batch, paper_matrix):
    """
    Compute cosine similarity between a batch of patent vectors and all paper vectors.
    Inputs: patent_batch (batch_size x dim), paper_matrix (num_papers x dim)
    Output: similarity matrix (batch_size x num_papers)
    """
    return cosine_similarity(patent_batch, paper_matrix)

def apply_threshold(sim_matrix, threshold):
    """
    Zero out similarities below the given threshold.
    """
    return np.where(sim_matrix >= threshold, sim_matrix, 0)

def get_rank_matrix(sim_matrix):
    """
    For each row, assign ranks to columns (1 = highest similarity).
    Input: sim_matrix (batch_size x num_papers)
    Output: rank_matrix (batch_size x num_papers)
    """
    rank_matrix = np.zeros_like(sim_matrix, dtype=int)
    for i in range(sim_matrix.shape[0]):
        order = np.argsort(-sim_matrix[i, :])
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, sim_matrix.shape[1] + 1)
        rank_matrix[i, :] = ranks
    return rank_matrix

def apply_wrrf(rank_func, rank_sol, rank_app, w_func=1, w_sol=1, w_app=1, k=60):
    """
    Weighted Reciprocal Rank Fusion (WRRF) to combine ranks from different fields.
    """
    return (w_func / (k + rank_func) + w_sol / (k + rank_sol) + w_app / (k + rank_app))

def count_topn_sdgs(combined_score, paper_sdg_labels, top_n):
    """
    For each patent, sum the SDG labels of the top-N scoring papers.
    Inputs: combined_score (batch_size x num_papers), paper_sdg_labels (num_papers x num_sdgs)
    Output: sdg_counts (batch_size x num_sdgs)
    """
    num_patents, _ = combined_score.shape
    num_sdgs = paper_sdg_labels.shape[1]
    sdg_counts = np.zeros((num_patents, num_sdgs))
    for i in range(num_patents):
        top_indices = np.argsort(-combined_score[i, :])[:top_n]
        for idx in top_indices:
            sdg_counts[i] += paper_sdg_labels[idx]
    return sdg_counts

def normalize_counts(count_matrix):
    """
    Normalize each row to sum to 1 (avoid division by zero).
    """
    sums = count_matrix.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return count_matrix / sums

def generate_sdg_vectors(
    dataset_path,
    functions_patent_path,
    solutions_patent_path,
    applications_patent_path,
    functions_paper_path,
    solutions_paper_path,
    applications_paper_path,
    paper_sdg_labels_path,
    output_path,
    batch_size=5000,
    sem_threshold_func=0.2604440815587794,
    sem_threshold_sol=0.16463215087216873,
    sem_threshold_app=0.8727727292417202,
    top_n=30
):

    data_train = pd.read_csv(dataset_path)
    functions_patent_train = np.load(functions_patent_path)
    solutions_patent_train = np.load(solutions_patent_path)
    applications_patent_train = np.load(applications_patent_path)
    functions_paper = np.load(functions_paper_path)
    solutions_paper = np.load(solutions_paper_path)
    applications_paper = np.load(applications_paper_path)
    paper_sdg_labels = np.load(paper_sdg_labels_path)

    # Check alignment
    assert len(data_train) == functions_patent_train.shape[0], "Dataset and patent embeddings must be aligned in size and order."

    num_patents = functions_patent_train.shape[0]
    num_sdgs = paper_sdg_labels.shape[1]
    total_sdg_counts = np.zeros((num_patents, num_sdgs))

    for start in tqdm(range(0, num_patents, batch_size)):
        end = min(start + batch_size, num_patents)

        batch_func_patent = functions_patent_train[start:end]
        batch_sol_patent = solutions_patent_train[start:end]
        batch_app_patent = applications_patent_train[start:end]

        func_sim = compute_similarity_matrix(batch_func_patent, functions_paper)
        sol_sim = compute_similarity_matrix(batch_sol_patent, solutions_paper)
        app_sim = compute_similarity_matrix(batch_app_patent, applications_paper)

        func_sim_thr = apply_threshold(func_sim, sem_threshold_func)
        sol_sim_thr = apply_threshold(sol_sim, sem_threshold_sol)
        app_sim_thr = apply_threshold(app_sim, sem_threshold_app)

        rank_func = get_rank_matrix(func_sim_thr)
        rank_sol = get_rank_matrix(sol_sim_thr)
        rank_app = get_rank_matrix(app_sim_thr)

        combined_score = apply_wrrf(rank_func, rank_sol, rank_app, 1, 1, 1)
        batch_sdg_counts = count_topn_sdgs(combined_score, paper_sdg_labels, top_n)

        total_sdg_counts[start:end, :] = batch_sdg_counts

    # Normalize counts to get soft distributions
    normalized_sdg_matrix = normalize_counts(total_sdg_counts)

    # Add to DataFrame and save
    data_train['sdg_vector'] = total_sdg_counts.tolist()
    data_train['sdg_vector_norm'] = normalized_sdg_matrix.tolist()
    data_train.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

def get_aligned_train_val_split(aligned_df, test_size=0.2, random_seed=42, sdg_vector_col='SDG_vector'):
    """
    Stratified multi-label split for the aligned dataset, using the SDG_vector column (binary vector per patent).
    Returns train and validation DataFrames, aligned for downstream use.
    This matches the logic used in classifier.py and data_alignment.py.
    """
    X_matrix = aligned_df.index.values.reshape(-1, 1)
    y_matrix = np.vstack(aligned_df[sdg_vector_col].apply(lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else np.array(eval(x))))
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X_matrix, y_matrix, test_size=test_size
    )
    train_df = aligned_df[aligned_df.index.isin(X_train.reshape(-1,).tolist())].reset_index(drop=True)
    val_df = aligned_df[aligned_df.index.isin(X_val.reshape(-1,).tolist())].reset_index(drop=True)
    return train_df, val_df