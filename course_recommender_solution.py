"""
Course Recommender Solution

Standalone Python version of the original analysis notebook.

Configuration (environment variables, no values hardcoded):
    RATINGS_URL  - set to your data file location
    GENRE_URL  - set to your data file location
    COURSES_URL  - set to your data file location
"""

import os

# ---- Configuration -------------------------------------------------
# Values are read from the environment so nothing is hardcoded.
RATINGS_URL = os.environ.get("RATINGS_URL", "")  # set to your data file location
GENRE_URL = os.environ.get("GENRE_URL", "")  # set to your data file location
COURSES_URL = os.environ.get("COURSES_URL", "")  # set to your data file location

# ---- Course Recommender System ----------------------------------------
# Scenario: an online-learning (MOOC) platform wants to recommend relevant courses to learners. This is a Proof-of-Concept comparing multiple content-based and collaborative-filtering models via offline evaluation.
# Pipeline: data understanding -> EDA -> BoW feature engineering + course similarity -> content-based recommenders (similarity, user-profile, clustering) -> collaborative filtering (KNN, NMF, neural embeddings) -> evaluation & comparison.

# ---- 0. Setup and data loading -----------------------------------------
# - ratings.csv - user / item / rating (enrollment mode: 2/3 = audit/complete, up to 5)
# - course_genre.csv - one-hot genre matrix per course
# - course_processed.csv - course id, title, description

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

COURSE_URL = COURSES_URL

ratings = pd.read_csv(RATINGS_URL)
course_genre = pd.read_csv(GENRE_URL)
courses = pd.read_csv(COURSE_URL)

print("ratings:", ratings.shape, "| course_genre:", course_genre.shape, "| courses:", courses.shape)
print(ratings.head())

# ---- 1. Exploratory Data Analysis (EDA) --------------------------------

print("Unique users   :", ratings["user"].nunique())
print("Unique courses :", ratings["item"].nunique())
print("Interactions   :", len(ratings))
print("Rating values  :", sorted(ratings["rating"].unique()))

fig, ax = plt.subplots(1, 2, figsize=(12,4))
sns.countplot(x="rating", data=ratings, ax=ax[0])
ax[0].set_title("Rating (enrollment mode) distribution")
per_user = ratings.groupby("user").size()
sns.histplot(per_user, bins=30, ax=ax[1])
ax[1].set_title("Courses enrolled per user")
ax[1].set_xlabel("number of courses")
plt.tight_layout(); plt.show()
print("Avg courses/user:", round(per_user.mean(), 2))

# Top 10 most popular courses
top = ratings["item"].value_counts().head(10).rename_axis("item").reset_index(name="enrollments")
top = top.merge(courses[["COURSE_ID","TITLE"]], left_on="item", right_on="COURSE_ID", how="left")
plt.figure(figsize=(9,4))
sns.barplot(y="TITLE", x="enrollments", data=top, color="royalblue")
plt.title("Top 10 courses by enrollment"); plt.tight_layout(); plt.show()
top[["TITLE","enrollments"]]

# Course count by genre
genre_cols = [c for c in course_genre.columns if c not in ("COURSE_ID","TITLE")]
genre_counts = course_genre[genre_cols].sum().sort_values(ascending=False)
plt.figure(figsize=(9,4))
genre_counts.plot.bar(color="seagreen")
plt.title("Number of courses per genre"); plt.ylabel("count"); plt.tight_layout(); plt.show()
print(genre_counts)

# Word cloud of course titles (optional)
try:
    from wordcloud import WordCloud
    text = " ".join(courses["TITLE"].astype(str).tolist())
    wc = WordCloud(width=900, height=350, background_color="white").generate(text)
    plt.figure(figsize=(11,4)); plt.imshow(wc); plt.axis("off")
    plt.title("Course title word cloud"); plt.show()
except Exception as e:
    print("wordcloud not installed (pip install wordcloud) - skipped:", e)

# ---- 2. Feature Engineering - Bag of Words + Course Similarity ---------
# We combine each course title and description, vectorize with TF-IDF (a weighted BoW), and compute the cosine-similarity matrix between all courses. This underpins the content-based similarity recommender.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

courses["text"] = courses["TITLE"].fillna("") + " " + courses["DESCRIPTION"].fillna("")
tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
X_text = tfidf.fit_transform(courses["text"])
print("BoW/TF-IDF matrix:", X_text.shape)

sim = cosine_similarity(X_text)
sim_df = pd.DataFrame(sim, index=courses["COURSE_ID"].values, columns=courses["COURSE_ID"].values)
print("Course-course similarity matrix:", sim_df.shape)

# ---- 3. Content-based Recommender #1 - Course Similarity ---------------
# Recommend courses whose BoW content is most similar to the courses a learner has already taken.

def recommend_by_similarity(user_id, top_n=10, threshold=0.3):
    enrolled = ratings[ratings["user"] == user_id]["item"].tolist()
    scores = {}
    for c in enrolled:
        if c in sim_df.index:
            col = sim_df[c]
            for other, sc in col.items():
                if other not in enrolled and sc >= threshold:
                    scores[other] = max(scores.get(other, 0.0), sc)
    return pd.Series(scores).sort_values(ascending=False).head(top_n)

sample_user = ratings["user"].iloc[0]
recs = recommend_by_similarity(sample_user).to_frame("similarity")
recs = recs.merge(courses.set_index("COURSE_ID")[["TITLE"]], left_index=True, right_index=True, how="left")
print("Similarity-based recommendations for user", sample_user)
print(recs)

# ---- 4. Content-based Recommender #2 - User Profile vs Course Genres ----
# Build a genre-preference vector for each learner (genre vectors of enrolled courses, weighted by rating), then score every unseen course by the dot product with that profile.

merged = ratings.merge(course_genre, left_on="item", right_on="COURSE_ID", how="inner")
weighted = merged.copy()
for g in genre_cols:
    weighted[g] = weighted[g] * weighted["rating"]
user_profiles = weighted.groupby("user")[genre_cols].sum()
genre_matrix = course_genre.set_index("COURSE_ID")[genre_cols]
print("user_profiles:", user_profiles.shape)

def recommend_by_profile(user_id, top_n=10):
    if user_id not in user_profiles.index:
        return pd.Series(dtype=float)
    profile = user_profiles.loc[user_id].values
    scores = genre_matrix.values.dot(profile)
    s = pd.Series(scores, index=genre_matrix.index)
    enrolled = set(ratings[ratings["user"] == user_id]["item"])
    s = s.drop(index=[i for i in enrolled if i in s.index], errors="ignore")
    return s.sort_values(ascending=False).head(top_n)

recs = recommend_by_profile(sample_user).to_frame("profile_score")
recs = recs.merge(courses.set_index("COURSE_ID")[["TITLE"]], left_index=True, right_index=True, how="left")
print("Profile-based recommendations for user", sample_user)
print(recs)

# ---- 5. Content-based Recommender #3 - Clustering (PCA + K-Means) ------
# Cluster learners by their genre profile (standardized, PCA-reduced), then recommend the most popular unseen courses within a learner's cluster.

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

Xp = StandardScaler().fit_transform(user_profiles.values)
pca = PCA(n_components=min(9, Xp.shape[1]), random_state=42)
Xr = pca.fit_transform(Xp)
print("PCA cumulative explained variance:", round(pca.explained_variance_ratio_.sum(), 3))

km = KMeans(n_clusters=20, random_state=42, n_init=10)
clusters = pd.Series(km.fit_predict(Xr), index=user_profiles.index, name="cluster")
ratings_c = ratings.merge(clusters, left_on="user", right_index=True, how="inner")

def recommend_by_cluster(user_id, top_n=10):
    if user_id not in clusters.index:
        return pd.Series(dtype=int)
    cl = clusters.loc[user_id]
    members = ratings_c[ratings_c["cluster"] == cl]
    pop = members["item"].value_counts()
    enrolled = set(ratings[ratings["user"] == user_id]["item"])
    pop = pop.drop(index=[i for i in enrolled if i in pop.index], errors="ignore")
    return pop.head(top_n)

recs = recommend_by_cluster(sample_user).to_frame("cluster_popularity")
recs = recs.merge(courses.set_index("COURSE_ID")[["TITLE"]], left_index=True, right_index=True, how="left")
print("Cluster-based recommendations for user", sample_user)
print(recs)

# ---- 6. Collaborative Filtering - KNN and NMF (Surprise) ---------------
# Matrix-factorization / neighborhood models trained on the user-item rating matrix, evaluated by RMSE. (Install once: pip install scikit-surprise.)

cf_results = {}
try:
    from surprise import Dataset, Reader, KNNBasic, NMF, accuracy
    from surprise.model_selection import train_test_split as surprise_split

    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
    data = Dataset.load_from_df(ratings[["user", "item", "rating"]], reader)
    trainset, testset = surprise_split(data, test_size=0.25, random_state=42)

    knn = KNNBasic(sim_options={"name": "cosine", "user_based": False})
    knn.fit(trainset)
    cf_results["KNN (item-based)"] = accuracy.rmse(knn.test(testset), verbose=False)

    nmf = NMF(n_factors=15, random_state=42)
    nmf.fit(trainset)
    cf_results["NMF"] = accuracy.rmse(nmf.test(testset), verbose=False)

    for k, v in cf_results.items():
        print(k, "RMSE:", round(v, 4))
except Exception as e:
    print("scikit-surprise not available -> run: pip install scikit-surprise")
    print("Error:", e)

# ---- 7. Collaborative Filtering - Neural Network Embeddings (Keras) ----
# A simple neural recommender: learn user and item embeddings, concatenate, and predict the rating (regression). Evaluated by RMSE. (Requires TensorFlow.)

try:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split as sk_split

    u_ids = ratings["user"].astype("category").cat.codes.values
    i_ids = ratings["item"].astype("category").cat.codes.values
    y = ratings["rating"].values.astype("float32")
    n_users, n_items = int(u_ids.max()) + 1, int(i_ids.max()) + 1

    Xu_tr, Xu_te, Xi_tr, Xi_te, y_tr, y_te = sk_split(
        u_ids, i_ids, y, test_size=0.25, random_state=42)

    emb = 16
    uin = tf.keras.Input(shape=(1,)); iin = tf.keras.Input(shape=(1,))
    ue = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(n_users, emb)(uin))
    ie = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(n_items, emb)(iin))
    x = tf.keras.layers.Concatenate()([ue, ie])
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model([uin, iin], out)
    model.compile(optimizer="adam", loss="mse")
    model.fit([Xu_tr, Xi_tr], y_tr, validation_split=0.1, epochs=5, batch_size=256, verbose=0)

    pred = model.predict([Xu_te, Xi_te], verbose=0).ravel()
    nn_rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
    cf_results["NN embeddings"] = nn_rmse
    print("NN embedding RMSE:", round(nn_rmse, 4))
except Exception as e:
    print("TensorFlow not available -> NN model skipped. Error:", e)

# ---- 8. Evaluation & Model Comparison ----------------------------------
# Compare the collaborative-filtering models by RMSE (lower is better). Content-based models are compared qualitatively (coverage / cold-start handling) since they do not predict a numeric rating.

if cf_results:
    comp = pd.Series(cf_results).sort_values()
    print("Collaborative-filtering RMSE comparison (lower is better):")
    print(comp.to_frame("RMSE"))
    plt.figure(figsize=(6,4))
    comp.plot.bar(color="royalblue")
    plt.ylabel("RMSE"); plt.title("CF model comparison"); plt.tight_layout(); plt.show()
    print("Best CF model:", comp.idxmin(), "with RMSE", round(comp.min(), 4))
else:
    print("No CF results yet - install scikit-surprise / tensorflow and re-run cells 6-7.")

# ---- 9. Top-N Ranking Evaluation (implicit feedback) -------------------
# Rating magnitude carries little signal (Section 8), so we evaluate the recommenders the correct way for implicit feedback: hide 20% of each learner's enrolled courses, recommend the top-K from the rest, and measure Precision@K / Recall@K / HitRate@K. (This cell may take ~1-2 minutes.)

from collections import defaultdict
import math
rng = np.random.default_rng(42)

user_items = ratings.groupby("user")["item"].apply(lambda s: list(dict.fromkeys(s)))
train_sets, test_sets = {}, {}
for u, its in user_items.items():
    if len(its) < 5:
        train_sets[u] = set(its); test_sets[u] = set()
    else:
        idx = rng.permutation(len(its))
        n_test = max(1, int(0.2 * len(its)))
        arr = np.array(its, dtype=object)
        test_sets[u] = set(arr[idx[:n_test]]); train_sets[u] = set(arr[idx[n_test:]])
eval_users = [u for u in user_items.index if test_sets[u]]
print("Evaluation users:", len(eval_users))

pop = defaultdict(int); cnt = defaultdict(int); co = defaultdict(lambda: defaultdict(int))
for u, s in train_sets.items():
    s = list(s)
    for i in s:
        pop[i] += 1; cnt[i] += 1
    for a in range(len(s)):
        for b in range(a + 1, len(s)):
            co[s[a]][s[b]] += 1; co[s[b]][s[a]] += 1
pop_rank = [i for i, _ in sorted(pop.items(), key=lambda x: -x[1])]
all_items = list(pop.keys())

gcols = [c for c in course_genre.columns if c not in ("COURSE_ID", "TITLE")]
gvec = {row["COURSE_ID"]: row[gcols].values.astype(float) for _, row in course_genre.iterrows()}

def rec_pop(u, k):
    tr = train_sets[u]; out = []
    for i in pop_rank:
        if i not in tr:
            out.append(i)
        if len(out) >= k:
            break
    return out

def rec_cf(u, k):
    tr = train_sets[u]; sc = defaultdict(float)
    for i in tr:
        for j, c in co[i].items():
            if j not in tr:
                sc[j] += c / math.sqrt(cnt[i] * cnt[j])
    return [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])[:k]]

def rec_content(u, k):
    tr = train_sets[u]; prof = np.zeros(len(gcols))
    for i in tr:
        if i in gvec:
            prof += gvec[i]
    sc = [(j, float(prof.dot(gvec[j]))) for j in all_items if j not in tr and j in gvec]
    return [i for i, _ in sorted(sc, key=lambda x: -x[1])[:k]]

def evaluate(rec_fn, k=10):
    P = R = H = 0; n = 0
    for u in eval_users:
        rel = test_sets[u]; recs = rec_fn(u, k)
        hits = len(set(recs) & rel)
        P += hits / k; R += hits / len(rel); H += 1 if hits > 0 else 0; n += 1
    return P / n, R / n, H / n

K = 10
rows = []
for name, fn in [("Popularity", rec_pop), ("Content (genre)", rec_content), ("Item-based CF", rec_cf)]:
    p, r, h = evaluate(fn, K)
    rows.append({"Model": name, ("Precision@%d" % K): round(p, 4),
                 ("Recall@%d" % K): round(r, 4), ("HitRate@%d" % K): round(h, 4)})
topn_results = pd.DataFrame(rows).sort_values(("Recall@%d" % K), ascending=False).reset_index(drop=True)
print(topn_results.to_string(index=False))

plt.figure(figsize=(6,4))
plt.bar(topn_results["Model"], topn_results["Recall@%d" % K], color="steelblue")
plt.ylabel("Recall@%d" % K)
plt.title("Top-N ranking - Recall@10 (higher is better)")
plt.xticks(rotation=15); plt.tight_layout(); plt.show()
print("Best ranker:", topn_results.iloc[0]["Model"])

# ---- 10. Conclusion & Recommendations ----------------------------------
# Findings
# - EDA: enrollments are dominated by a handful of popular courses (e.g., Python, Data Science, ML); most learners take only a few courses -> a sparse user-item matrix.
# - Ratings are near-uniform (enrollment mode 3-5), so RMSE barely separates models - simple baselines match matrix factorization. RMSE is the wrong north-star here.
# - Top-N ranking (Section 9) is the meaningful evaluation: item-based collaborative filtering wins (highest Recall@10 / HitRate@10), well above the popularity and content baselines.
# - Content-based models remain valuable for cold-start (new courses/learners) and recommendation diversity.
# Recommendation for the platform
# - Deploy a hybrid ranking system: item-based collaborative filtering for active learners + content-based recommenders for cold-start.
# - Select and monitor the production model with top-N metrics (precision@k, recall@k, hit-rate), not rating RMSE.
# - Next steps: hyperparameter tuning, richer text features, dedicated implicit-feedback models (e.g. ALS/BPR), and online A/B testing.
# (A Streamlit app - app.py - lets stakeholders pick a model and generate recommendations interactively.)
