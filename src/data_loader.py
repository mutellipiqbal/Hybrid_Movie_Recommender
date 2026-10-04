"""
data_loader.py
---------------
Loads and lightly cleans the MovieLens ml-latest-small dataset
(Harper & Konstan, 2015) for use by the content-based, collaborative,
and hybrid recommenders.

Dataset files expected in ../data/:
    movies.csv   (movieId, title, genres)
    ratings.csv  (userId, movieId, rating, timestamp)
    tags.csv     (userId, movieId, tag, timestamp)
    links.csv    (movieId, imdbId, tmdbId)
"""
import os
import re
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _extract_year(title: str):
    """Pull a 4-digit release year out of a MovieLens title like
    'Toy Story (1995)'. Returns None if no year is present."""
    match = re.search(r"\((\d{4})\)\s*$", title.strip())
    return int(match.group(1)) if match else None


def load_movies() -> pd.DataFrame:
    """Load movies.csv and add a clean_title / year / genre_list column."""
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    movies["year"] = movies["title"].apply(_extract_year)
    movies["clean_title"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True
    )
    movies["genre_list"] = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    return movies


def load_ratings() -> pd.DataFrame:
    """Load ratings.csv (userId, movieId, rating, timestamp)."""
    return pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))


def load_tags() -> pd.DataFrame:
    """Load tags.csv (userId, movieId, tag, timestamp)."""
    return pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))


def build_tag_corpus(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every free-text tag applied to a movie into one string,
    and combine it with the pipe-separated genre list into a single
    'content_soup' text field that the TF-IDF vectorizer can consume.
    This mirrors the metadata-combination step used by the credited
    foundation project (see README.md 'Credits'), substituting
    MovieLens genres + community tags for its TMDB overview/keyword
    fields since this dataset has no plot synopses.
    """
    tag_strings = (
        tags.dropna(subset=["tag"])
        .groupby("movieId")["tag"]
        .apply(lambda s: " ".join(str(t).lower().replace(" ", "_") for t in s))
        .rename("tag_soup")
    )
    df = movies.merge(tag_strings, on="movieId", how="left")
    df["tag_soup"] = df["tag_soup"].fillna("")
    genre_soup = df["genre_list"].apply(lambda gl: " ".join(g.lower() for g in gl))
    df["content_soup"] = (genre_soup + " " + df["tag_soup"]).str.strip()
    return df


def popularity_table(ratings: pd.DataFrame) -> pd.DataFrame:
    """Per-movie rating count and mean rating, used for the popularity
    prior that helps mitigate the cold-start / long-tail noise problem
    IBM (2024) describes."""
    agg = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    agg.columns = ["rating_count", "rating_mean"]
    return agg
