"""
Core semantic matching and optimization for SDG-patent-paper association.

This script takes as input:
- Embedding matrices for patents and papers for each extracted category, i.e function, solutions, applications. (from create_embeddings_paeter.py).
- paper_sdg_labels: a binary matrix (num_papers x num_sdgs) indicating which SDGs are associated with each paper.
- citation_sdg_counts: a matrix (num_patents x num_sdgs) indicating the SDG distribution of NPL citations for each patent (target for optimization).

Key logic:
- Patent vectors must be split into two parts (e.g., train/val or test/val) for validation and optimization.
- For each patent, semantic similarity is computed to all papers using cosine similarity for each field (function, solution, application).
- Similarity matrices are thresholded, ranked, and combined using a reciprocal rank fusion (RRF) scheme.
- For each patent, the top-N most similar papers are selected, and their SDG labels are aggregated to predict the SDG distribution for the patent.
- The predicted SDG distribution is compared to the true citation-based SDG distribution using a positive-only MSE loss.
- Optuna is used to optimize the thresholds and top-N parameter to minimize the loss.

Inputs:
- functions_patent_val, solutions_patent_val, applications_patent_val: Validation split of patent embeddings (numpy arrays).
- functions_paper, solutions_paper, applications_paper: Paper embeddings (numpy arrays).
- citation_sdg_counts_val: Validation split of citation SDG counts (numpy array).
- paper_sdg_labels: Paper-to-SDG association matrix (numpy array).
- The script expects the patent vectors to be split (e.g., train/val) so that validation can be performed.

The script can be easily modified in order to accept only one full_text embeddings matrix,
and of course using only one threshold as described in the paper for the baseline comparison.
"""

import numpy as np
from tqdm import tqdm
import optuna
import matplotlib.pyplot as plt
import json
from sklearn.metrics.pairwise import cosine_similarity


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


def positive_only_mse(pred_counts, true_counts):
    """
    Compute MSE only for SDG slots where the true count is positive.
    """
    mse = 0
    count = 0
    for i in range(true_counts.shape[0]):
        for j in range(true_counts.shape[1]):
            if true_counts[i, j] > 0:
                mse += (pred_counts[i, j] - true_counts[i, j]) ** 2
                count += 1
    return mse / count if count > 0 else np.nan

# --- Optuna objective with preallocated batching ---
def optuna_objective(trial, functions_patent, solutions_patent, applications_patent,
                     functions_paper, solutions_paper, applications_paper,
                     citation_sdg_counts, paper_sdg_labels, batch_size=5000, sdg=False):
    """
    Optuna objective for hyperparameter optimization.
    Runs in batches over the validation split of patent vectors (_val),
    computes semantic similarity, aggregates SDG predictions, and returns positive-only MSE.
    """

    # starting ranges of hyperparameters
    sem_threshold_func = trial.suggest_float('sem_threshold_func', 0.1, 0.9)
    sem_threshold_sol = trial.suggest_float('sem_threshold_sol', 0.1, 0.9)
    sem_threshold_app = trial.suggest_float('sem_threshold_app', 0.1, 0.9)
    top_n = trial.suggest_int('top_n', 5, 30)

    num_patents = functions_patent.shape[0]
    num_sdgs = paper_sdg_labels.shape[1]
    total_sdg_counts = np.zeros((num_patents, num_sdgs))

    for start in tqdm(range(0, num_patents, batch_size)):
        end = min(start + batch_size, num_patents)
        batch_func_patent = functions_patent[start:end]
        batch_sol_patent = solutions_patent[start:end]
        batch_app_patent = applications_patent[start:end]

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

    pred_sdg_norm = normalize_counts(total_sdg_counts)
    true_sdg_norm = normalize_counts(citation_sdg_counts)
    score = positive_only_mse(pred_sdg_norm, true_sdg_norm)

    print(f"Trial {trial.number} | MSE: {score:.5f} | sem_threshold: {(sem_threshold_func, sem_threshold_sol, sem_threshold_app)}, top_n: {top_n}")
    return score


def run_optuna_study(functions_patent_val, solutions_patent_val, applications_patent_val,
                     functions_paper, solutions_paper, applications_paper,
                     citation_sdg_counts_val, paper_sdg_labels, n_trials=100, output_json="best_params.json"):
    """
    Run Optuna hyperparameter optimization for semantic SDG prediction.
    Inputs: *_val are validation splits of patent vectors and citation SDG counts.
    Saves best parameters to output_json and shows optimization plots.
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: optuna_objective(trial, functions_patent_val, solutions_patent_val,
                                                 applications_patent_val, functions_paper, solutions_paper,
                                                 applications_paper, citation_sdg_counts_val, paper_sdg_labels),
                   n_trials=n_trials)

    print("\nBest trial")
    best_trial = study.best_trial
    print(f"  Value (MSE): {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    with open(output_json, 'w') as f:
        json.dump(best_trial.params, f, indent=4)
        print(f"Best parameters saved to {output_json}")

    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization History")
    plt.show()

    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Parameter Importance")
    plt.show()

    return study