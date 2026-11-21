"""
Course Recommender - Streamlit app 
================================================================
Interactive demo of the content-based and collaborative-filtering
recommenders built in course_recommender_solution.py.

Run:
    pip install streamlit pandas numpy scikit-learn
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
RATINGS_URL = BASE + "IBMSkillsNetwork-ML0321EN-Coursera/labs/v2/module_3/ratings.csv"
GENRE_URL = BASE + "IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
COURSE_URL = BASE + "IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv"


@st.cache_data
def load_data():
    ratings = pd.read_csv(RATINGS_URL)
    course_genre = pd.read_csv(GENRE_URL)
    courses = pd.read_csv(COURSE_URL)
    return ratings, course_genre, courses


@st.cache_data
def build_similarity(courses):
    text = courses["TITLE"].fillna("") + " " + courses["DESCRIPTION"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
    X = tfidf.fit_transform(text)
    sim = cosine_similarity(X)
    return pd.DataFrame(sim, index=courses["COURSE_ID"].values,
                        columns=courses["COURSE_ID"].values)


@st.cache_data
def build_profiles(ratings, course_genre):
    genre_cols = [c for c in course_genre.columns if c not in ("COURSE_ID", "TITLE")]
    merged = ratings.merge(course_genre, left_on="item", right_on="COURSE_ID", how="inner")
    for g in genre_cols:
        merged[g] = merged[g] * merged["rating"]
    profiles = merged.groupby("user")[genre_cols].sum()
    genre_matrix = course_genre.set_index("COURSE_ID")[genre_cols]
    return profiles, genre_matrix, genre_cols


def title_of(courses, ids):
    m = courses.set_index("COURSE_ID")["TITLE"]
    return [m.get(i, i) for i in ids]


def fetch_similarity_score(user_id, ratings, sim_df, top_n, threshold=0.3):
    enrolled = ratings[ratings["user"] == user_id]["item"].tolist()
    scores = {}
    for c in enrolled:
        if c in sim_df.index:
            for other, sc in sim_df[c].items():
                if other not in enrolled and sc >= threshold:
                    scores[other] = max(scores.get(other, 0.0), sc)
    return pd.Series(scores).sort_values(ascending=False).head(top_n)


def fetch_profile_score(user_id, ratings, profiles, genre_matrix, top_n):
    if user_id not in profiles.index:
        return pd.Series(dtype=float)
    scores = genre_matrix.values.dot(profiles.loc[user_id].values)
    s = pd.Series(scores, index=genre_matrix.index)
    enrolled = set(ratings[ratings["user"] == user_id]["item"])
    s = s.drop(index=[i for i in enrolled if i in s.index], errors="ignore")
    return s.sort_values(ascending=False).head(top_n)


def main():
    st.set_page_config(page_title="Course Recommender", layout="wide")
    st.title("Course Recommender")
    st.caption("PoC: content-based and collaborative-filtering recommenders")

    ratings, course_genre, courses = load_data()
    sim_df = build_similarity(courses)
    profiles, genre_matrix, genre_cols = build_profiles(ratings, course_genre)

    with st.sidebar:
        st.header("Settings")
        model = st.selectbox("Recommendation model",
                             ["Course similarity (content)",
                              "User profile / genre (content)"])
        users = ratings["user"].unique()
        user_id = st.selectbox("Learner (user id)", sorted(users)[:500])
        top_n = st.slider("Number of recommendations", 5, 20, 10)

    enrolled = ratings[ratings["user"] == user_id]["item"].tolist()
    st.subheader("Already enrolled")
    st.write(pd.DataFrame({"course": title_of(courses, enrolled)}))

    if model.startswith("Course similarity"):
        rec_score = fetch_similarity_score(user_id, ratings, sim_df, top_n)
        score_name = "similarity"
    else:
        rec_score = fetch_profile_score(user_id, ratings, profiles, genre_matrix, top_n)
        score_name = "profile_score"

    st.subheader("Recommended courses")
    if len(rec_score) == 0:
        st.info("No recommendations for this learner with the current settings.")
    else:
        out = pd.DataFrame({"course": title_of(courses, rec_score.index),
                            score_name: np.round(rec_score.values, 3)})
        st.dataframe(out, width="stretch")


if __name__ == "__main__":
    main()
