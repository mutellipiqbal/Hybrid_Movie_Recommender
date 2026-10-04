"""
Quick end-to-end smoke test for the recommendation pipeline (not a full
pytest suite -- run directly with `python tests/test_pipeline.py` from
the project root). Confirms the data loads, every component builds
without error, and a hybrid recommendation + explanation comes back for
a couple of realistic queries.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_movies, load_ratings, load_tags, build_tag_corpus, popularity_table
from src.content_based import ContentRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.query_parser import TitleIndex


def main():
    print("Loading data...")
    movies = load_movies()
    ratings = load_ratings()
    tags = load_tags()
    print(f"  {len(movies)} movies, {len(ratings)} ratings, {len(tags)} tags")

    movies_soup = build_tag_corpus(movies, tags)
    pop = popularity_table(ratings)

    print("Building content-based recommender...")
    content_rec = ContentRecommender(movies_soup)

    print("Building collaborative recommender (TruncatedSVD)...")
    collab_rec = CollaborativeRecommender(ratings, n_factors=20)

    hybrid = HybridRecommender(movies, content_rec, collab_rec, pop)
    index = TitleIndex(movies)

    test_queries = [
        "I loved Toy Story and Jumanji, what should I watch next?",
        "Something like The Matrix",
        "I really enjoyed Forrest Gump",
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print("QUERY:", q)
        liked_ids = index.extract_movie_ids(q)
        print("Matched to:", [index.title_for(m) for m in liked_ids])
        if not liked_ids:
            print("  (no match -- skipping)")
            continue
        recs = hybrid.recommend(liked_ids, top_n=5)
        for mid, blended, c, cf, p in recs:
            print(f"  -> {hybrid.explain(liked_ids, mid, c, cf)}  [blend={blended:.3f} content={c:.3f} collab={cf:.3f} pop={p:.3f}]")

    print("\n" + "=" * 70)
    print("Smoke test complete -- no exceptions raised.")


if __name__ == "__main__":
    main()
