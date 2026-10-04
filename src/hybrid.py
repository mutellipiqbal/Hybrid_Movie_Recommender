"""
hybrid.py
---------
Blends the content-based and collaborative recommenders into a single
ranked list, applies a popularity prior to damp down noisy long-tail
items, and generates a short natural-language explanation for each
recommendation. This weighted-blend hybridization follows the general
approach described by Burke (2002) for combining recommendation
techniques, and is the central extension this project adds on top of
the (purely content-based) foundation project.
"""
import numpy as np
import pandas as pd


class HybridRecommender:
    def __init__(self, movies: pd.DataFrame, content_rec, collab_rec, pop_table: pd.DataFrame,
                 content_weight: float = 0.55, collab_weight: float = 0.35, pop_weight: float = 0.10):
        self.movies = movies.set_index("movieId")
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.pop_table = pop_table
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.pop_weight = pop_weight

        counts = pop_table["rating_count"]
        self._pop_norm = (np.log1p(counts) / np.log1p(counts.max())) if len(counts) else counts

    def _pop_score(self, movie_id: int) -> float:
        return float(self._pop_norm.get(movie_id, 0.0))

    def recommend(self, liked_movie_ids, top_n: int = 8, candidate_pool: int = 60):
        """Core hybrid ranking. liked_movie_ids may contain one or more
        movieIds (a single click, or several movies named in a chat
        query). Content and collaborative candidate pools are generated
        separately, merged, re-scored with the weighted blend below, and
        the popularity prior is folded in so that very obscure titles
        with a lucky TF-IDF match don't dominate the top of the list."""
        content_hits = dict(self.content_rec.similar_to_profile(liked_movie_ids, top_n=candidate_pool))

        collab_ids = [m for m in liked_movie_ids if self.collab_rec.has_movie(m)]
        collab_hits = dict(self.collab_rec.similar_to_profile(collab_ids, top_n=candidate_pool)) if collab_ids else {}

        candidate_ids = set(content_hits) | set(collab_hits)
        candidate_ids -= set(liked_movie_ids)

        scored = []
        for mid in candidate_ids:
            c_score = content_hits.get(mid, 0.0)
            cf_score = collab_hits.get(mid, 0.0)
            p_score = self._pop_score(mid)
            blended = (
                self.content_weight * c_score
                + self.collab_weight * cf_score
                + self.pop_weight * p_score
            )
            scored.append((mid, blended, c_score, cf_score, p_score))

        scored.sort(key=lambda t: -t[1])
        return scored[:top_n]

    def explain(self, liked_movie_ids, movie_id: int, c_score: float, cf_score: float) -> str:
        """Build a short, human-readable justification for recommending
        movie_id, citing whichever signal (content overlap vs. rating
        pattern) contributed more -- the kind of transparency Nielsen
        Norman Group (2026) flags as a trust-building design quality for
        conversational AI (see the companion virtual-assistants report)."""
        title = self.movies.loc[movie_id, "title"] if movie_id in self.movies.index else f"movie #{movie_id}"
        reasons = []

        if c_score > 0:
            shared_terms = set()
            for liked_id in liked_movie_ids:
                shared_terms.update(self.content_rec.top_matching_terms(liked_id, movie_id, top_n=3))
            shared_terms.discard("")
            if shared_terms:
                reasons.append(f"shares themes/genres ({', '.join(sorted(shared_terms)[:3])}) with what you liked")
            else:
                reasons.append("has similar genre/tag content to what you liked")

        if cf_score > 0.3:
            reasons.append("was also rated highly by users with similar taste")

        if not reasons:
            reasons.append("is a popular, well-rated title that's broadly similar to your request")

        return f"{title} -- " + "; and ".join(reasons) + "."
