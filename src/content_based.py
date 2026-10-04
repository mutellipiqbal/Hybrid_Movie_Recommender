"""
content_based.py
-----------------
Content-based filtering component.

This is the piece of the system most directly adapted from the credited
foundation project, sudhanshudwivedii/Movie_Recommendation_System
(MIT License), which vectorizes movie text metadata with TF-IDF and
ranks candidates by cosine similarity. See README.md 'Credits' for the
full citation and a description of what was changed.

Modification from the original: the foundation project vectorizes a
TMDB plot-overview + genre + keyword "soup" from movies_metadata.csv.
This project's dataset (MovieLens ml-latest-small) has no plot
synopses, so the text soup here is built from pipe-separated genres
plus the free-text community tags in tags.csv instead (see
data_loader.build_tag_corpus). The vectorizer settings (unigrams +
bigrams, English stop words) follow the same approach as the
foundation project, adapted to this smaller, sparser vocabulary.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pandas as pd


class ContentRecommender:
    def __init__(self, movies_with_soup: pd.DataFrame):
        self.movies = movies_with_soup.reset_index(drop=True)
        self._id_to_row = {mid: i for i, mid in enumerate(self.movies["movieId"])}

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["content_soup"])

    def similar_to_movie(self, movie_id: int, top_n: int = 10):
        """Return [(movieId, similarity_score), ...] for the top_n movies
        most textually similar to movie_id, via cosine similarity on the
        TF-IDF vectors (linear_kernel is equivalent to cosine similarity
        here because TF-IDF vectors are L2-normalized by scikit-learn)."""
        if movie_id not in self._id_to_row:
            return []
        row = self._id_to_row[movie_id]
        sims = linear_kernel(self.tfidf_matrix[row], self.tfidf_matrix).flatten()
        ranked = np.argsort(-sims)
        results = []
        for i in ranked:
            cand_id = int(self.movies.loc[i, "movieId"])
            if cand_id == movie_id:
                continue
            results.append((cand_id, float(sims[i])))
            if len(results) >= top_n:
                break
        return results

    def similar_to_profile(self, liked_movie_ids, top_n: int = 10):
        """Average the TF-IDF vectors of several liked movies into a
        single taste profile, then rank all other movies by similarity
        to that profile. This supports the multi-movie 'I liked X and Y'
        conversational queries the base project's single-title lookup
        does not handle."""
        rows = [self._id_to_row[m] for m in liked_movie_ids if m in self._id_to_row]
        if not rows:
            return []
        profile = self.tfidf_matrix[rows].mean(axis=0)
        profile = np.asarray(profile)
        sims = linear_kernel(profile, self.tfidf_matrix).flatten()
        ranked = np.argsort(-sims)
        results = []
        liked_set = set(liked_movie_ids)
        for i in ranked:
            cand_id = int(self.movies.loc[i, "movieId"])
            if cand_id in liked_set:
                continue
            results.append((cand_id, float(sims[i])))
            if len(results) >= top_n:
                break
        return results

    def top_matching_terms(self, movie_id_a: int, movie_id_b: int, top_n: int = 3):
        """Return the shared vocabulary terms that contribute most to the
        similarity between two movies, used to generate the
        human-readable 'why this was recommended' explanation."""
        if movie_id_a not in self._id_to_row or movie_id_b not in self._id_to_row:
            return []
        vocab = np.array(self.vectorizer.get_feature_names_out())
        vec_a = self.tfidf_matrix[self._id_to_row[movie_id_a]].toarray().flatten()
        vec_b = self.tfidf_matrix[self._id_to_row[movie_id_b]].toarray().flatten()
        contribution = vec_a * vec_b
        if contribution.sum() == 0:
            return []
        top_idx = np.argsort(-contribution)[:top_n]
        return [vocab[i] for i in top_idx if contribution[i] > 0]
