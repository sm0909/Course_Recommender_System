"""
Course Recommender - quantitative evaluation
============================================

One script that answers four questions about the recommender PoC with numbers:

1. How big is the data?              -> users, courses, interactions, sparsity, rating spread
2. What is the baseline RMSE?         -> global-mean predictor and user/item-bias predictor
3. What is the final (best) RMSE?     -> KNN, NMF, SVD (Surprise) and a neural embedding model
4. How good are the top-K lists?      -> Precision@K, Recall@K, NDCG@K, HitRate@K, coverage
                                         for popularity, content-based, clustering, CF, NMF
                                         and a hybrid, on held-out enrolments

Everything is seeded, downloads are cached in ./data, and results land in ./results:
    metrics.json, rmse_comparison.csv, ranking_metrics.csv, *.png, RESULTS.md

Run:   python3 evaluate_recommenders.py           (~3-5 minutes on a laptop CPU)
"""

import json
import os
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # write PNGs, never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF as SkNMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

SEED = 42
K_VALUES = (5, 10, 20)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"

BASE = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
RATINGS_URL = os.environ.get("RATINGS_URL", BASE + "IBMSkillsNetwork-ML0321EN-Coursera/labs/v2/module_3/ratings.csv")
GENRE_URL = os.environ.get("GENRE_URL", BASE + "IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv")
COURSES_URL = os.environ.get("COURSES_URL", BASE + "IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv")


# ----------------------------------------------------------------------------- data
def fetch(url):
    """Download once into ./data and reuse the local copy afterwards."""
    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / url.rsplit("/", 1)[-1]
    if not target.exists():
        print("downloading", target.name)
        urllib.request.urlretrieve(url, target)
    return pd.read_csv(target)


def describe_dataset(ratings):
    users, items = ratings["user"].nunique(), ratings["item"].nunique()
    per_user = ratings.groupby("user").size()
    return {
        "interactions": int(len(ratings)),
        "users": int(users),
        "courses": int(items),
        "sparsity_pct": round(100 * (1 - len(ratings) / (users * items)), 2),
        "rating_distribution": {int(k): int(v) for k, v in ratings["rating"].value_counts().sort_index().items()},
        "courses_per_user_mean": round(float(per_user.mean()), 2),
        "courses_per_user_median": float(per_user.median()),
        "users_with_5plus_courses": int((per_user >= 5).sum()),
    }


# ----------------------------------------------------------------------------- RMSE part
def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def neural_embedding_rmse(train, test, epochs=5, dim=16):
    """
    PyTorch port of the notebook's Keras model: user + course embeddings -> Dense(32) -> rating.
    (Same layer sizes, optimiser, epochs and batch size; PyTorch so TensorFlow isn't required.)
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    users = {u: i for i, u in enumerate(train["user"].unique())}
    items = {c: i for i, c in enumerate(train["item"].unique())}
    test = test[test["user"].isin(users) & test["item"].isin(items)]   # can't score unseen ids

    def tensors(frame):
        return (torch.tensor(frame["user"].map(users).values), torch.tensor(frame["item"].map(items).values),
                torch.tensor(frame["rating"].values, dtype=torch.float32))

    class EmbeddingRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.user = nn.Embedding(len(users), dim)
            self.item = nn.Embedding(len(items), dim)
            self.head = nn.Sequential(nn.Linear(2 * dim, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, u, i):
            return self.head(torch.cat([self.user(u), self.item(i)], dim=1)).squeeze(1)

    model = EmbeddingRegressor()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    u, i, y = tensors(train)
    for _ in range(epochs):
        order = torch.randperm(len(y))
        for start in range(0, len(y), 256):
            batch = order[start:start + 256]
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(model(u[batch], i[batch]), y[batch])
            loss.backward()
            optimiser.step()
    tu, ti, ty = tensors(test)
    with torch.no_grad():
        pred = model(tu, ti).clamp(3, 5).numpy()
    return rmse(pred, ty.numpy()), mae(pred, ty.numpy())


def rating_prediction_results(ratings):
    """Train on 75 % of the ratings, report RMSE/MAE on the other 25 % (same split as the notebook)."""
    from surprise import (SVD, BaselineOnly, Dataset, KNNBasic, KNNWithMeans, NMF, Reader)
    from surprise.model_selection import train_test_split

    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
    data = Dataset.load_from_df(ratings[["user", "item", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=SEED)
    truth = np.array([r for (_, _, r) in testset])

    rows = []
    # Baseline 1 - always predict the average rating. Any real model has to beat this.
    mean_pred = np.full_like(truth, trainset.global_mean, dtype=float)
    rows.append(("Global mean (baseline)", rmse(mean_pred, truth), mae(mean_pred, truth)))

    models = {
        "User + course bias (baseline)": BaselineOnly(verbose=False),
        "KNN item-based (cosine)": KNNBasic(sim_options={"name": "cosine", "user_based": False}, verbose=False),
        "KNN with means (item-based)": KNNWithMeans(sim_options={"name": "pearson_baseline", "user_based": False},
                                                     verbose=False),
        "NMF (15 factors)": NMF(n_factors=15, random_state=SEED),
        "SVD (matrix factorisation)": SVD(random_state=SEED),
    }
    for name, model in models.items():
        started = time.time()
        model.fit(trainset)
        predictions = np.array([p.est for p in model.test(testset)])
        rows.append((name, rmse(predictions, truth), mae(predictions, truth)))
        print(f"  {name:<32} RMSE {rows[-1][1]:.4f}  ({time.time() - started:.0f}s)")

    # Neural embeddings on the same 75/25 split (rebuilt as DataFrames).
    train_df = pd.DataFrame(trainset.build_testset(), columns=["user", "item", "rating"])
    test_df = pd.DataFrame(testset, columns=["user", "item", "rating"])
    nn_rmse, nn_mae = neural_embedding_rmse(train_df, test_df)
    rows.append(("Neural embeddings (PyTorch)", nn_rmse, nn_mae))
    print(f"  {'Neural embeddings (PyTorch)':<32} RMSE {nn_rmse:.4f}")

    table = pd.DataFrame(rows, columns=["model", "rmse", "mae"]).sort_values("rmse").reset_index(drop=True)
    return table, trainset.n_ratings, len(testset)


# ----------------------------------------------------------------------------- top-K ranking part
def holdout_split(ratings, holdout_fraction=0.2, min_items=5):
    """
    Hide 20 % of each learner's courses (learners with >= 5 courses). Recommenders only see the
    rest; a good recommender should put the hidden courses near the top of its list.
    """
    rng = np.random.default_rng(SEED)
    train_rows, test_rows = [], []
    for user, group in ratings.groupby("user"):
        items = group["item"].tolist()
        if len(items) < min_items:
            train_rows += [(user, i) for i in items]
            continue
        order = rng.permutation(len(items))
        n_test = max(1, int(holdout_fraction * len(items)))
        test_rows += [(user, items[j]) for j in order[:n_test]]
        train_rows += [(user, items[j]) for j in order[n_test:]]
    return pd.DataFrame(train_rows, columns=["user", "item"]), pd.DataFrame(test_rows, columns=["user", "item"])


def to_matrix(frame, user_index, item_index):
    matrix = np.zeros((len(user_index), len(item_index)), dtype=np.float32)
    matrix[frame["user"].map(user_index).values, frame["item"].map(item_index).values] = 1.0
    return matrix


def cosine_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalised = matrix / np.clip(norms, 1e-9, None)
    similarity = normalised @ normalised.T
    np.fill_diagonal(similarity, 0.0)
    return similarity


def build_scorers(train_matrix, items, course_genre, courses):
    """Each scorer returns a (users x courses) score matrix; higher = recommend first."""
    genre_cols = [c for c in course_genre.columns if c not in ("COURSE_ID", "TITLE")]
    genres = course_genre.set_index("COURSE_ID").reindex(items)[genre_cols].fillna(0).values.astype(np.float32)

    text = courses.set_index("COURSE_ID").reindex(items)
    text = (text["TITLE"].fillna("") + " " + text["DESCRIPTION"].fillna("")).values
    tfidf = TfidfVectorizer(stop_words="english", max_features=2000).fit_transform(text).toarray()

    popularity = train_matrix.sum(axis=0)
    co_counts = train_matrix.T @ train_matrix                      # how often two courses are taken together
    counts = np.clip(np.diag(co_counts), 1, None)
    item_cf = co_counts / np.sqrt(np.outer(counts, counts))
    np.fill_diagonal(item_cf, 0.0)

    scorers = {}
    scorers["Popularity (baseline)"] = lambda m: np.tile(popularity, (m.shape[0], 1))
    scorers["Content: genre profile"] = lambda m: (m @ genres) @ genres.T
    bow_sim = cosine_rows(tfidf)
    scorers["Content: BoW course similarity"] = lambda m: m @ bow_sim

    # Clustering: group learners by genre profile (PCA + K-Means), recommend what's popular in the group.
    profiles = StandardScaler().fit_transform(train_matrix @ genres)
    reduced = PCA(n_components=min(9, profiles.shape[1]), random_state=SEED).fit_transform(profiles)
    labels = KMeans(n_clusters=20, random_state=SEED, n_init=10).fit_predict(reduced)
    cluster_pop = np.vstack([train_matrix[labels == c].sum(axis=0) for c in range(20)])
    scorers["Clustering (PCA + K-Means)"] = lambda m, labels=labels: cluster_pop[labels]

    scorers["Item-based CF (co-occurrence)"] = lambda m: m @ item_cf

    nmf = SkNMF(n_components=32, init="nndsvda", random_state=SEED, max_iter=300)
    user_factors = nmf.fit_transform(train_matrix)
    scorers["NMF (implicit, 32 factors)"] = lambda m: user_factors @ nmf.components_

    def hybrid(m):
        # Blend collaborative and content signals after scaling each to [0, 1] per learner.
        def scale(s):
            low, high = s.min(axis=1, keepdims=True), s.max(axis=1, keepdims=True)
            return (s - low) / np.clip(high - low, 1e-9, None)
        return 0.8 * scale(m @ item_cf) + 0.2 * scale((m @ genres) @ genres.T)
    scorers["Hybrid (0.8 CF + 0.2 content)"] = hybrid
    return scorers


def ranking_metrics(scores, train_matrix, test_matrix, k):
    scores = np.where(train_matrix > 0, -np.inf, scores)            # never recommend what they've taken
    top_k = np.argsort(-scores, axis=1)[:, :k]
    hits = np.take_along_axis(test_matrix, top_k, axis=1)           # 1 where a hidden course was recommended
    relevant = test_matrix.sum(axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(axis=1)
    ideal = np.array([discounts[: int(min(r, k))].sum() for r in relevant])
    return {
        f"precision@{k}": float(hits.sum(axis=1).mean() / k),
        f"recall@{k}": float((hits.sum(axis=1) / relevant).mean()),
        f"ndcg@{k}": float((dcg / ideal).mean()),
        f"hit_rate@{k}": float((hits.sum(axis=1) > 0).mean()),
        f"coverage@{k}": float(len(np.unique(top_k)) / scores.shape[1]),
    }


def ranking_results(ratings, course_genre, courses):
    train, test = holdout_split(ratings)
    items = sorted(ratings["item"].unique())
    item_index = {c: i for i, c in enumerate(items)}
    user_index = {u: i for i, u in enumerate(sorted(ratings["user"].unique()))}
    train_matrix = to_matrix(train, user_index, item_index)
    test_matrix = to_matrix(test, user_index, item_index)

    eval_users = test_matrix.sum(axis=1) > 0
    scorers = build_scorers(train_matrix, items, course_genre, courses)
    rows = []
    for name, scorer in scorers.items():
        scores = scorer(train_matrix)[eval_users]
        row = {"model": name}
        for k in K_VALUES:
            row.update(ranking_metrics(scores, train_matrix[eval_users], test_matrix[eval_users], k))
        rows.append(row)
        print(f"  {name:<34} P@10 {row['precision@10']:.4f}  R@10 {row['recall@10']:.4f}  NDCG@10 {row['ndcg@10']:.4f}")
    table = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    split_info = {"evaluation_users": int(eval_users.sum()), "held_out_enrolments": int(test_matrix.sum()),
                  "training_enrolments": int(train_matrix.sum())}
    return table, split_info


# ----------------------------------------------------------------------------- reporting
def save_charts(rmse_table, rank_table, ratings):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(rmse_table["model"], rmse_table["rmse"], color="#4C72B0")
    ax.set_xlim(rmse_table["rmse"].min() * 0.95, rmse_table["rmse"].max() * 1.02)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE on held-out ratings (lower is better)")
    ax.set_title("Rating prediction: baselines vs models")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "rmse_comparison.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics = ["precision@10", "recall@10", "ndcg@10"]
    x = np.arange(len(rank_table))
    for offset, metric in enumerate(metrics):
        ax.bar(x + (offset - 1) * 0.27, rank_table[metric], width=0.27, label=metric)
    ax.set_xticks(x); ax.set_xticklabels(rank_table["model"], rotation=25, ha="right", fontsize=8)
    ax.set_title("Top-10 ranking quality on held-out enrolments"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "ranking_metrics.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ratings["rating"].value_counts().sort_index().plot.bar(ax=ax, color="#55A868")
    ax.set_title("Rating distribution"); ax.set_xlabel("rating"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "rating_distribution.png", dpi=130); plt.close(fig)


def write_markdown(summary, rmse_table, rank_table):
    lines = ["# Course Recommender - evaluation results", "",
             "Generated by `evaluate_recommenders.py` (seed 42).", "", "## Dataset", ""]
    for key, value in summary["dataset"].items():
        lines.append(f"- **{key}**: {value}")
    lines += ["", "## Rating prediction (75/25 split)", "", rmse_table.round(4).to_markdown(index=False), "",
              f"- Baseline RMSE (global mean): **{summary['rmse']['baseline_rmse']:.4f}**",
              f"- Final RMSE (best learned model, {summary['rmse']['best_model']}): "
              f"**{summary['rmse']['final_rmse']:.4f}**",
              f"- Change vs baseline: **{summary['rmse']['improvement_pct']:.2f}%** "
              "(positive = lower RMSE than the baseline)",
              "- The ratings are spread almost evenly over 3/4/5, so they carry very little signal and no "
              "model beats the global mean by much. Top-K ranking below is what actually separates the models.",
              "",
              "## Top-K ranking (20% of each learner's courses held out)", "",
              rank_table.round(4).to_markdown(index=False), ""]
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)
    ratings, course_genre, courses = fetch(RATINGS_URL), fetch(GENRE_URL), fetch(COURSES_URL)

    print("\n[1/3] Dataset")
    dataset = describe_dataset(ratings)
    print(json.dumps(dataset, indent=2))

    print("\n[2/3] Rating prediction (RMSE)")
    rmse_table, train_size, test_size = rating_prediction_results(ratings)
    baseline = float(rmse_table.loc[rmse_table["model"] == "Global mean (baseline)", "rmse"].iloc[0])
    # "Final" = the best *learned* model (the two baselines are only there to beat).
    best = rmse_table[~rmse_table["model"].str.contains("baseline")].iloc[0]
    rmse_summary = {"baseline_rmse": baseline, "bias_baseline_rmse":
                    float(rmse_table.loc[rmse_table["model"] == "User + course bias (baseline)", "rmse"].iloc[0]),
                    "best_model": best["model"], "final_rmse": float(best["rmse"]),
                    "improvement_pct": round(100 * (baseline - best["rmse"]) / baseline, 2),
                    "train_ratings": train_size, "test_ratings": test_size}
    rmse_table.to_csv(RESULTS_DIR / "rmse_comparison.csv", index=False)

    print("\n[3/3] Top-K ranking (Precision / Recall / NDCG)")
    rank_table, split_info = ranking_results(ratings, course_genre, courses)
    rank_table.to_csv(RESULTS_DIR / "ranking_metrics.csv", index=False)
    pop = rank_table[rank_table["model"] == "Popularity (baseline)"].iloc[0]
    top = rank_table.iloc[0]
    ranking_summary = {**split_info, "best_ranker": top["model"],
                       **{f"best_{m}": round(float(top[m]), 4) for m in
                          ("precision@10", "recall@10", "ndcg@10", "hit_rate@10")},
                       "ndcg@10_lift_over_popularity_pct": round(100 * (top["ndcg@10"] - pop["ndcg@10"]) / pop["ndcg@10"], 2)}

    summary = {"dataset": dataset, "rmse": rmse_summary, "ranking": ranking_summary}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    save_charts(rmse_table, rank_table, ratings)
    write_markdown(summary, rmse_table, rank_table)
    print("\nSummary:\n" + json.dumps({"rmse": rmse_summary, "ranking": ranking_summary}, indent=2))
    print(f"\nAll outputs written to {RESULTS_DIR.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
