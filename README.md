# Hybrid Movie Recommender (Chatbot Interface)

A content-based + collaborative-filtering hybrid movie recommendation
system with a conversational (chat) interface, built for MSAI-631-A01
(Artificial Intelligence for Human-Computer Interaction), University of
the Cumberlands. Full write-up of the design process, theory, and
challenges is in the accompanying APA report
(`Recommendation_System_Report.docx`); this README documents the code.

## What it does

Tell it a movie or two you like in plain English --

> "I loved Toy Story and The Matrix, what should I watch next?"

-- and it replies with five ranked recommendations, each with a short,
human-readable reason (shared genres/tags, and/or "users with similar
taste also rated this highly").

## Credits

This project's **content-based filtering component**
(`src/content_based.py`) is adapted from the approach used in
**sudhanshudwivedii/Movie_Recommendation_System**
(<https://github.com/sudhanshudwivedii/Movie_Recommendation_System>,
MIT License): vectorize movie text metadata with TF-IDF (unigrams +
bigrams, English stop words removed) and rank candidates by cosine
similarity. No code was copied verbatim -- this is an independent
implementation of the same technique, adapted to a different dataset
(see "What was changed" below).

**Dataset:** F. Maxwell Harper and Joseph A. Konstan. 2015. *The
MovieLens Datasets: History and Context.* ACM Transactions on
Interactive Intelligent Systems (TiiS) 5, 4: 19:1-19:19.
<https://doi.org/10.1145/2827872>. The `ml-latest-small` dataset
(9,742 movies, 100,836 ratings from 610 users) is redistributed here
under GroupLens's research-use license; see `data/MOVIELENS_README.txt`
for the original terms and citation.

**Theoretical grounding:** the collaborative-filtering component
implements the latent-factor / matrix-factorization approach described
in Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization
techniques for recommender systems. *Computer*, 42(8), 42-49. The
blending strategy follows Burke, R. (2002). Hybrid recommender
systems: Survey and experiments. *User Modeling and User-Adapted
Interaction*, 12(4), 331-370. Full citations are in the report.

## What was changed / added relative to the foundation project

The foundation project is a single Jupyter notebook that does
content-based filtering only, against TMDB plot-overview metadata, with
no deployed interface (its README lists a Streamlit app as a future
item). This project:

1. **Swaps the metadata source.** MovieLens has no plot synopses, so
   the TF-IDF "soup" is built from pipe-separated genres plus
   community-submitted tags (`src/data_loader.build_tag_corpus`)
   instead of TMDB overviews/keywords.
2. **Adds a collaborative-filtering component from scratch**
   (`src/collaborative.py`): the foundation project has no concept of
   other users' ratings at all. This project factorizes the 610-user x
   9,742-movie rating matrix with truncated SVD and ranks movies by
   latent-factor similarity -- an operationalization of Koren et al.'s
   (2009) matrix-factorization theory.
3. **Adds a weighted hybrid blender with a popularity prior**
   (`src/hybrid.py`) that combines the content and collaborative
   scores and dampens noisy long-tail matches, per Burke (2002).
4. **Adds a natural-language query parser** (`src/query_parser.py`)
   so the user can name movies in a free-text sentence instead of
   picking one exact title from a dropdown, including handling
   MovieLens's "Matrix, The" article-at-the-end title convention.
5. **Adds a from-scratch chat interface** (`app.py`, Gradio
   `ChatInterface`) and a plain-language explanation generator for
   each recommendation (`HybridRecommender.explain`).

## Architecture

```
app.py                     Gradio chat interface (entry point)
src/
  data_loader.py           Load & clean MovieLens CSVs; build TF-IDF text soup
  content_based.py         TF-IDF + cosine similarity (adapted from credited project)
  collaborative.py         TruncatedSVD latent-factor collaborative filtering
  hybrid.py                Weighted blend + popularity prior + explanations
  query_parser.py          Free-text -> movieId matching for the chat interface
data/                      MovieLens ml-latest-small CSVs + original license/README
tests/test_pipeline.py     End-to-end smoke test (no network/API calls)
```

## Running it

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python tests/test_pipeline.py     # optional: quick console smoke test
python app.py                     # launches the chat UI at http://127.0.0.1:7860
```

No API keys, accounts, or internet access are required at run time --
the dataset ships in `data/` and everything runs locally.

## Known limitations

See the report's "Challenges and Mitigation Strategies" section for the
full discussion. Briefly: the query parser is rule-based rather than
an LLM, so it can miss paraphrased or misspelled titles, and a small
blocklist of ambiguous common-word titles (e.g., "Next", "Up", "It")
is excluded from single-word matching to avoid false positives like
treating "what should I watch next" as a reference to the film *Next*
(2007). The dataset caps out in 2018, so it cannot recommend anything
more recent.
