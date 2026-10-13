from localsearch import pipeline
from localsearch.index import Index
from localsearch.pipeline import Searcher

FILLER = " It was recorded in the ship's log by the officers that afternoon."

BOOK = "\n\n".join([
    "CHAPTER XVII",
    "We arrived at the islands in September after a long passage." + FILLER,
    "The giant tortoises of these islands are very large and slow, and they eat cactus." + FILLER,
    "They drink great quantities of water, walking for days to reach the springs." + FILLER,
    "The sailors repaired the rigging and painted the hull before supper." + FILLER,
    "CHAPTER XVIII",
    "The tortoises were also hunted for their meat by passing whalers." + FILLER,
])


class FakeModel:
    """Routes on the prompt text, so each pipeline step can be scripted."""

    def __init__(self, keywords="tortoise, shell", continues="NO", relevant="YES", squeeze=None):
        self.keywords = keywords
        self.continues = continues
        self.relevant = relevant
        self.squeeze = squeeze
        self.prompts = []

    def __call__(self, prompt, max_tokens):
        self.prompts.append(prompt)
        if prompt.startswith("List 3 to 6"):
            return self.keywords
        if "continue talking" in prompt:
            return self.continues
        if "Answer YES or NO" in prompt:
            return self.relevant
        if "Copy the sentences" in prompt:
            if self.squeeze is not None:
                return self.squeeze
            text = prompt.split("Text:\n", 1)[1]
            return text.split(".")[0] + "."
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")


def make_index(tmp_path, text=BOOK):
    folder = tmp_path / "books"
    folder.mkdir()
    (folder / "voyage.txt").write_text(text)
    idx = Index(folder, tmp_path / "i.db")
    idx.refresh()
    return idx


# --- pure helpers ----------------------------------------------------------


def test_question_terms_keep_quoted_phrases_and_drop_stopwords():
    assert pipeline.question_terms('tell me about "giant tortoises" and their food') == ["giant tortoises", "food"]


def test_parse_keywords_survives_model_mess():
    reply = "Keywords: 1. Tortoise\n2. \"Galapagos\"\n- shell,\nthis is a very long rambling sentence not a keyword"
    assert pipeline.parse_keywords(reply) == ["tortoise", "galapagos", "shell"]


def test_parse_yes_no():
    assert pipeline.parse_yes_no("YES.") is True
    assert pipeline.parse_yes_no("No, it does not") is False
    assert pipeline.parse_yes_no("I think the paragraph...") is None
    assert pipeline.parse_yes_no("yesterday nobody") is None  # whole words only


def test_verify_extract_returns_source_wording_and_drops_inventions():
    source = "The tortoise eats cactus every day. The ship sailed at dawn. Whalers hunted them."
    reply = "- the tortoise eats cactus each day\nTortoises can fly to the moon on Tuesdays\nNONE"
    assert pipeline.verify_extract(reply, source) == ["The tortoise eats cactus every day."]


def test_keyword_sentences_picks_best_matches_in_order():
    source = "Ships sail. The tortoise eats cactus. Nothing here. A tortoise shell is hard."
    roots = pipeline._roots(["tortoise", "shell"])
    assert pipeline.keyword_sentences(source, roots, limit=2) == ["The tortoise eats cactus.", "A tortoise shell is hard."]


def test_headings_and_continuations_are_decided_without_a_model():
    roots = pipeline._roots(["tortoise"])
    assert pipeline.continues_cheaply("CHAPTER XVIII", roots) is False
    assert pipeline.continues_cheaply("The tortoises drink water.", roots) is True
    assert pipeline.continues_cheaply("However, the springs were far away.", roots) is True
    assert pipeline.continues_cheaply("The sailors painted the hull before supper.", roots) is None


def test_junk_detection():
    assert pipeline.looks_like_junk("THE BARBARIANS BREAK THE EMPIRE INTO EAST AND WEST 227 XL.")
    assert pipeline.looks_like_junk("Tortoises, 27, 28, 101, 344; shells, 12, 19")
    assert not pipeline.looks_like_junk("The tortoise weighed 200 pounds and was very old.")


def test_merge_joins_touching_passages_in_same_file():
    P = pipeline.Passage
    merged = pipeline.merge_passages([P(1, "a", 3, 3, ["d"], -2.0), P(1, "a", 1, 2, ["b", "c"], -1.0), P(2, "b", 4, 4, ["x"], -3.0)])
    assert [(p.path, p.start, p.end, p.paragraphs, p.score) for p in merged] == [
        ("b", 4, 4, ["x"], -3.0),
        ("a", 1, 3, ["b", "c", "d"], -2.0),
    ]


# --- the whole pipeline ----------------------------------------------------


def test_no_llm_mode_finds_and_extracts_without_any_model(tmp_path):
    g = Searcher(make_index(tmp_path), ask=None).gather("giant tortoises")
    labels = [p.label for p in g.passages]
    assert labels and all(lbl.startswith("voyage.txt") for lbl in labels)
    assert any("giant tortoises" in s for p in g.passages for s in p.extract)


def test_growth_follows_continuation_and_stops_at_heading(tmp_path):
    model = FakeModel()
    s = Searcher(make_index(tmp_path), ask=model, max_grow=5)
    g = s.gather("giant tortoises")
    first = next(p for p in g.passages if "giant tortoises" in p.text)
    # "They drink..." continues by rule; sailors paragraph needs the model (NO); heading never joins
    assert (first.start, first.end) == (2, 3)
    assert any("continue talking" in pr and "sailors" in pr for pr in model.prompts)
    assert not any("CHAPTER" in p.text for p in g.passages)


def test_model_saying_continue_grows_the_passage(tmp_path):
    s = Searcher(make_index(tmp_path), ask=FakeModel(continues="YES"), max_grow=5)
    first = next(p for p in s.gather("giant tortoises").passages if "giant" in p.text)
    assert first.end == 4  # took the sailors paragraph, stopped at CHAPTER XVIII


def test_filter_drops_passages_the_model_rejects(tmp_path):
    g = Searcher(make_index(tmp_path), ask=FakeModel(relevant="NO")).gather("giant tortoises")
    assert g.passages == []


def test_garbage_relevance_answer_keeps_the_passage(tmp_path):
    g = Searcher(make_index(tmp_path), ask=FakeModel(relevant="well, maybe")).gather("giant tortoises")
    assert g.passages


def test_invented_squeeze_falls_back_to_keyword_sentences(tmp_path):
    model = FakeModel(squeeze="Tortoises are secretly excellent at chess and poetry.")
    g = Searcher(make_index(tmp_path), ask=model).gather("giant tortoises")
    all_sentences = [s for p in g.passages for s in p.extract]
    assert all_sentences
    assert not any("chess" in s for s in all_sentences)
    assert all(s in BOOK for s in all_sentences)  # every word came from the book


def test_model_that_raises_degrades_to_rules(tmp_path):
    def broken(prompt, max_tokens):
        raise RuntimeError("server went away")

    g = Searcher(make_index(tmp_path), ask=broken).gather("giant tortoises")
    assert g.passages  # filter keeps on unclear, squeeze falls back to keywords


def test_nothing_found(tmp_path):
    g = Searcher(make_index(tmp_path), ask=FakeModel(keywords="")).gather("quantum chromodynamics")
    assert g.passages == []


def test_tldr_prompt_numbers_the_notes():
    P = pipeline.Passage
    p = P(1, "a.txt", 0, 0, ["x"], -1.0, extract=["Fact one.", "Fact two."])
    prompt = pipeline.tldr_prompt("q?", [p])
    assert "[1] Fact one. Fact two." in prompt


def test_budget_keeps_best_passages_that_fit(tmp_path):
    P = pipeline.Passage
    big = [P(1, "a", i, i, ["x"], float(i), extract=["w" * 1500]) for i in range(10)]
    s = Searcher(make_index(tmp_path), context_size=2048)
    kept = s._fit_budget("q", big)
    assert 0 < len(kept) < 10
    assert [p.score for p in kept] == sorted(p.score for p in kept)
