"""
app.py
------
Conversational / chat-style front end for the hybrid movie
recommender, built with Gradio. This is the project's third
extension beyond the foundation project, which has no deployed
interface at all (only a notebook). Per the assignment, the system
can be accessed via "a Graphic User Interface (GUI), text prompt,
chatbot, or conversational/intelligent agent" -- this app is a
chatbot-style GUI: the user types a free-text message naming one or
more movies they like, and the assistant replies with a ranked,
explained list of recommendations, the same way a conversational
agent would.

Run locally with:
    source venv/bin/activate
    python app.py
then open the printed local URL (default http://127.0.0.1:7860).
"""
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no telemetry / version-check calls

import gradio as gr

from src.data_loader import load_movies, load_ratings, load_tags, build_tag_corpus, popularity_table
from src.content_based import ContentRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.query_parser import TitleIndex

print("Loading MovieLens data and building recommenders (this takes a few seconds)...")
_movies = load_movies()
_ratings = load_ratings()
_tags = load_tags()
_movies_soup = build_tag_corpus(_movies, _tags)
_pop = popularity_table(_ratings)

_content_rec = ContentRecommender(_movies_soup)
_collab_rec = CollaborativeRecommender(_ratings, n_factors=20)
_hybrid = HybridRecommender(_movies, _content_rec, _collab_rec, _pop)
_index = TitleIndex(_movies)
print("Ready.")

WELCOME = (
    "Hi! I'm a small movie-recommendation assistant. Tell me a movie or two you "
    "like -- for example *\"I loved Toy Story and The Matrix\"* or *\"something "
    "like Forrest Gump\"* -- and I'll suggest five more, with a short reason for "
    "each one. I only know the ~9,700 movies in the MovieLens ml-latest-small "
    "dataset (mostly pre-2018 titles), so very recent releases won't be in here."
)


def respond(message, history):
    liked_ids = _index.extract_movie_ids(message)

    if not liked_ids:
        return (
            "I couldn't recognize a movie title in that message. Could you name "
            "a specific movie, e.g. \"I liked Jurassic Park\"? (Note: this demo "
            "only knows movies released before 2018.)"
        )

    matched_titles = ", ".join(_index.title_for(m) for m in liked_ids)
    recs = _hybrid.recommend(liked_ids, top_n=5)

    if not recs:
        return f"I matched that to {matched_titles}, but couldn't find enough related titles to recommend from."

    lines = [f"Got it -- matched that to **{matched_titles}**. Here's what I'd try next:\n"]
    for rank, (mid, blended, c_score, cf_score, p_score) in enumerate(recs, start=1):
        lines.append(f"{rank}. {_hybrid.explain(liked_ids, mid, c_score, cf_score)}")
    return "\n".join(lines)


demo = gr.ChatInterface(
    fn=respond,
    title="Hybrid Movie Recommender (MovieLens ml-latest-small)",
    description=WELCOME,
    examples=[
        "I loved Toy Story and The Matrix, what should I watch next?",
        "Something like Forrest Gump",
        "I really enjoyed The Shawshank Redemption and Fight Club",
    ],
    analytics_enabled=False,
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
