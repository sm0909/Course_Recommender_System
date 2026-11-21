# Course Recommender : Using Content-based & Collaborative-filtering recommendations

A recommender system for an online-learning (MOOC) platform that suggests the next courses a learner
is likely to take. It builds and compares content-based and collaborative-filtering models, then
selects the best one with an offline evaluation that measures both rating prediction and the quality
of the top-K recommendation lists learners actually see.

## What I built

* **Feature engineering**: Bag-of-Words / TF-IDF vectors from course titles and descriptions, and a
  course-to-course cosine-similarity matrix.
* **Content-based recommenders**: course similarity, a learner-profile × course-genre scorer, and an
  unsupervised recommender that clusters learners (PCA + K-Means) and recommends what is popular in
  each cluster.
* **Collaborative-filtering recommenders**: item-based KNN, KNN-with-means, NMF and SVD (Surprise), a
  neural user/course embedding model, an item co-occurrence recommender, implicit-feedback NMF, and a
  hybrid that blends collaborative and content signals.
* **Evaluation harness** (`evaluate_recommenders.py`): dataset profiling, baseline vs model RMSE on the
  same split, and leave-20%-out top-K ranking metrics (Precision@K, Recall@K, NDCG@K, HitRate@K,
  catalogue coverage for K = 5/10/20) against a popularity baseline. The neural embedding model is
  implemented in PyTorch here, so the harness needs no TensorFlow.
* **Interactive demo** (`app.py`): a Streamlit app to pick a model and a learner and see recommendations.

## Repository layout

| File | Purpose |
|------|---------|
| `course_recommender_solution.py` | End-to-end analysis: EDA → features → content-based and collaborative recommenders → comparison (also available as the notebook `Course_Recommender_Solution.ipynb`) |
| `evaluate_recommenders.py` | Quantitative evaluation harness |
| `app.py` | Streamlit demo app |
| `run_evaluation.sh` | One-command setup + evaluation |
| `requirements.txt` | Dependencies for the analysis script and the app |
| `requirements-eval.txt` | Minimal dependencies for the evaluation harness |

## Datasets (downloaded automatically)

* `ratings.csv` - user / item / rating (enrolment mode)
* `course_genre.csv` - one-hot genre matrix per course
* `course_processed.csv` - course id, title, description

## Prerequisites

* Linux, macOS or WSL (Ubuntu) with **Python 3.10+**; about 2 GB of free RAM; internet access on the
  first run (~5 MB of data, cached in `data/`).

## Setup

```bash
git clone <this-repo> && cd <this-repo>
python3 -m venv .venv && source .venv/bin/activate      # optional; without it pip installs with --user
pip install -r requirements-eval.txt                     # evaluation harness
pip install -r requirements.txt                          # analysis script + Streamlit app
```

## How to run

### 1. Evaluate and compare the recommenders

```bash
./run_evaluation.sh                  # installs requirements-eval.txt, then runs the harness
python3 evaluate_recommenders.py     # same, without the install step (about 1 minute on CPU)
```

What it does, step by step:

1. Downloads the three CSV files into `data/` (first run only).
2. Profiles the dataset: learners, courses, enrolments, sparsity, rating distribution.
3. Splits the ratings 75/25 with a fixed seed and computes RMSE/MAE for two baselines (global mean,
   user + course bias) and for KNN, KNN-with-means, NMF, SVD and the neural embedding model.
4. Hides 20% of the courses of every learner with at least 5 enrolments, asks each recommender for
   its top-K courses, and computes Precision@K, Recall@K, NDCG@K, HitRate@K and coverage.
5. Writes the tables, JSON summary and charts listed under *Outputs*.

### 2. Run the full analysis

```bash
export RATINGS_URL="<url-or-path-to>/ratings.csv"
export GENRE_URL="<url-or-path-to>/course_genre.csv"
export COURSES_URL="<url-or-path-to>/course_processed.csv"
python3 course_recommender_solution.py       # or open Course_Recommender_Solution.ipynb and Run All
```

### 3. Launch the demo app

```bash
streamlit run app.py
```

## Configuration

| Variable | Used by | Meaning |
|----------|---------|---------|
| `RATINGS_URL`, `GENRE_URL`, `COURSES_URL` | `course_recommender_solution.py` (required), `evaluate_recommenders.py` (optional override) | Location of the three CSV files |

## Models implemented

| Task | Technique |
|------|-----------|
| EDA | distributions, popular courses, genre counts, word cloud |
| BoW features + course similarity | TF-IDF + cosine similarity |
| Content-based | course similarity |
| Content-based | user profile vs genres |
| Content-based (unsupervised) | PCA + K-Means clustering |
| Collaborative filtering | KNN, KNN-with-means, NMF, SVD (Surprise) |
| Collaborative filtering | neural-network embeddings; item co-occurrence; implicit NMF |
| Hybrid | weighted blend of collaborative and content scores |
| Evaluation | RMSE/MAE vs baselines; Precision/Recall/NDCG/HitRate@K and coverage |

## Outputs

| Path | Contents |
|------|----------|
| `results/RESULTS.md` | Dataset profile, RMSE table, top-K ranking table |
| `results/metrics.json` | All metrics in one file |
| `results/rmse_comparison.csv`, `results/ranking_metrics.csv` | Metric tables as CSV |
| `results/rmse_comparison.png`, `results/ranking_metrics.png`, `results/rating_distribution.png` | Charts |
| `results/run_log.txt` | Console log of the run |
| `data/` | Cached datasets (not committed) |

## Design notes

* Rating prediction and ranking are evaluated separately: a low RMSE does not guarantee good top-K
  lists, and learners only ever see the top of the list.
* Content-based models handle **cold-start** (new learners or courses) that collaborative models
  cannot, so the recommended production design is a **hybrid**.

## Data credits

Course enrolment, genre and description datasets — public sample datasets hosted by IBM Skills Network
on IBM Cloud Object Storage.
