"""
query_parser.py
----------------
Very small natural-language front end for the chatbot interface. This
is the project's second major extension beyond the foundation project,
which only exposes a single "pick one title from a list" lookup. Here
a free-text message such as

    "I loved Toy Story and The Matrix, what should I watch next?"

is scanned for movie titles the user mentions (matched against the
MovieLens title list), rather than requiring an exact dropdown
selection. This is intentionally a lightweight, rule-based parser --
not an LLM -- so the whole system can run locally with no API keys or
external calls; see the report's 'Challenges and Mitigation Strategies'
section for why this tradeoff was made and what it still gets wrong.

Two matching passes are used, in order:
  1. Contiguous word-sequence matching: every MovieLens title is
     checked against the normalized word list of the whole message to
     see if its own words appear in order and adjacent (handling
     MovieLens's "Matrix, The" article-at-the-end convention by also
     indexing the natural "The Matrix" word order). This is the
     primary pass and is what correctly pulls "Toy Story" and "The
     Matrix" out of a full sentence.
  2. Fuzzy difflib matching on comma/"and"-separated clauses, used only
     as a fallback for misspelled or slightly-off titles the exact pass
     missed.
"""
import difflib
import re
import pandas as pd

_LEADING_ARTICLES = (", the", ", a", ", an")

# Short, single-word movie titles that double as common English words
# (pronouns, prepositions, conversational filler). Matching these
# literally causes false positives like "next" in "what should I watch
# next?" being read as a reference to the film Next (2007). Excluded
# from single-word contiguous matching; see the report's "Challenges
# and Mitigation Strategies" section for the tradeoff this creates.
_AMBIGUOUS_SINGLE_WORD_TITLES = {
    "up", "it", "her", "him", "he", "she", "they", "them", "next", "me",
    "i", "we", "us", "this", "that", "them", "now", "here", "there",
}


class TitleIndex:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies
        self._key_to_id = {}      # normalized title text -> movieId
        self._key_words = {}      # normalized title text -> tuple(words)
        self._choices = []

        for _, row in movies.iterrows():
            mid = row["movieId"]
            clean = row["clean_title"]
            for key in self._title_keys(clean):
                norm = self._normalize(key)
                if norm and norm not in self._key_to_id:
                    self._key_to_id[norm] = mid
                    self._key_words[norm] = tuple(norm.split())
                    self._choices.append(norm)

        # Longest (most specific / least ambiguous) titles first, so a
        # match on "Toy Story 2" is preferred over the shorter "Toy Story"
        # when both would fit, and common short titles are tried last.
        self._choices.sort(key=lambda k: -len(k))

    @staticmethod
    def _title_keys(clean_title: str):
        """Return both the literal title and, if it ends in a
        MovieLens-style trailing article ('Matrix, The'), the natural
        reading order ('The Matrix') so users can type titles normally."""
        keys = [clean_title]
        lower = clean_title.lower()
        for suffix in _LEADING_ARTICLES:
            if lower.endswith(suffix):
                article = clean_title[-(len(suffix) - 2):]  # 'The' / 'A' / 'An'
                base = clean_title[: -len(suffix)]
                keys.append(f"{article} {base}")
        return keys

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _contiguous_matches(self, free_text: str):
        words = self._normalize(free_text).split()
        n = len(words)
        covered = [False] * n
        found = []  # list of (start_index, movieId), sorted by position at the end

        for key in self._choices:              # longest titles first
            kwords = self._key_words[key]
            klen = len(kwords)
            if klen == 1 and (len(kwords[0]) < 4 or kwords[0] in _AMBIGUOUS_SINGLE_WORD_TITLES):
                continue  # skip very short / common-word one-word titles to cut false positives
            for start in range(0, n - klen + 1):
                if any(covered[start:start + klen]):
                    continue
                if tuple(words[start:start + klen]) == kwords:
                    mid = self._key_to_id[key]
                    if mid not in {m for _, m in found}:
                        found.append((start, mid))
                    for i in range(start, start + klen):
                        covered[i] = True
                    break

        found.sort(key=lambda t: t[0])
        return [mid for _, mid in found]

    def _fuzzy_fallback(self, free_text: str, cutoff: float = 0.82):
        found = []
        pieces = re.split(r",| and |&|\n", free_text, flags=re.IGNORECASE)
        for piece in pieces:
            norm = self._normalize(piece)
            if len(norm) < 3:
                continue
            matches = difflib.get_close_matches(norm, self._choices, n=1, cutoff=cutoff)
            if matches:
                mid = self._key_to_id[matches[0]]
                if mid not in found:
                    found.append(mid)
        return found

    def extract_movie_ids(self, free_text: str, max_titles: int = 5):
        found = self._contiguous_matches(free_text)
        if not found:
            found = self._fuzzy_fallback(free_text)
        return found[:max_titles]

    def best_exact_or_close(self, title_guess: str, cutoff: float = 0.82):
        norm = self._normalize(title_guess)
        if norm in self._key_to_id:
            return self._key_to_id[norm]
        matches = difflib.get_close_matches(norm, self._choices, n=1, cutoff=cutoff)
        return self._key_to_id[matches[0]] if matches else None

    def title_for(self, movie_id: int) -> str:
        row = self.movies.loc[self.movies["movieId"] == movie_id]
        return row.iloc[0]["title"] if len(row) else f"movie #{movie_id}"
