"""
collaborative.py
-----------------
Collaborative filtering component, added to extend the foundation
project (which is purely content-based and has no notion of other
users' ratings at all).

Implements the latent-factor approach described by Koren, Bell, and
Volinsky (2009): the sparse user x item rating matrix is factorized
into a user-factor matrix P and an item-factor matrix Q via truncated
singular value decomposition (TruncatedSVD), such that the dot product
of a user's and item's latent vectors approximates the user's rating
for that item. Item-item similarity in the resulting latent space is
then used to find movies favored by users with similar taste profiles,
in the spirit of the model-based (as opposed to memory-based /
nearest-neighbor) collaborative filtering Koren et al. (2009) report
as outperforming classic neighborhood methods.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class CollaborativeRecommender:
    def __init__(self, ratings: pd.DataFrame, n_factors: int = 20, min_ratings: int = 3):
        # Keep only movies with at least `min_ratings` ratings so the
        # factor model isn't dominated by one-off noise.
        counts = ratings["movieId"].value_counts()
        keep_movies = counts[counts >= min_ratings].index
        ratings = ratings[ratings["movieId"].isin(keep_movies)]

        self.user_ids = sorted(ratings["userId"].unique())
        self.movie_ids = sorted(ratings["movieId"].unique())
        self._uid_to_row = {u: i for i, u in enumerate(self.user_ids)}
        self._mid_to_col = {m: i for i, m in enumerate(self.movie_ids)}
        self._col_to_mid = {i: m for m, i in self._mid_to_col.items()}

        rows = ratings["userId"].map(self._uid_to_row)
        cols = ratings["movieId"].map(self._mid_to_col)
        mat = csr_matrix(
            (ratings["rating"].astype(float), (rows, cols)),
            shape=(len(self.user_ids), len(self.movie_ids)),
        )

        n_factors = min(n_factors, min(mat.shape) - 1) if min(mat.shape) > 1 else 1
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_factors = self.svd.fit_transform(mat)          # P: users x factors
        self.item_factors = self.svd.components_.T               # Q: items x factors
        self.item_factors_norm = normalize(self.item_factors)     # for cosine similarity

    def similar_to_movie(self, movie_id: int, top_n: int = 10):
        """Return [(movieId, similarity_score), ...] for the movies whose
        latent-factor vectors (i.e., whose rating patterns across users)
        are most similar to movie_id's -- 'users who rated this highly
        also tended to rate these highly'."""
        if movie_id not in self._mid_to_col:
            return []
        col = self._mid_to_col[movie_id]
        target = self.item_factors_norm[col].reshape(1, -1)
        sims = (self.item_factors_norm @ target.T).flatten()
        ranked = np.argsort(-sims)
        results = []
        for i in ranked:
            cand_id = self._col_to_mid[i]
            if cand_id == movie_id:
                continue
            results.append((cand_id, float(sims[i])))
            if len(results) >= top_n:
                break
        return results

    def similar_to_profile(self, liked_movie_ids, top_n: int = 10):
        """Average the latent-factor vectors of several liked movies and
        rank all other movies by cosine similarity to that point in
        latent space -- the collaborative-filtering analogue of
        ContentRecommender.similar_to_profile."""
        cols = [self._mid_to_col[m] for m in liked_movie_ids if m in self._mid_to_col]
        if not cols:
            return []
        profile = normalize(self.item_factors[cols].mean(axis=0, keepdims=True))
        sims = (self.item_factors_norm @ profile.T).flatten()
        ranked = np.argsort(-sims)
        liked_set = set(liked_movie_ids)
        results = []
        for i in ranked:
            cand_id = self._col_to_mid[i]
            if cand_id in liked_set:
                continue
            results.append((cand_id, float(sims[i])))
            if len(results) >= top_n:
                break
        return results

    def has_movie(self, movie_id: int) -> bool:
        return movie_id in self._mid_to_col
