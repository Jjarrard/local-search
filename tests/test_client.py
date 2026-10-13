from localsearch.client import ThinkFilter, is_chat_model, strip_think


def feed_all(pieces):
    f = ThinkFilter()
    return "".join(f.feed(p) for p in pieces)


def test_strip_think_closed_and_unclosed():
    assert strip_think("<think>hmm</think>\nYES") == "YES"
    assert strip_think("<think>still going") == ""
    assert strip_think("NO") == "NO"


def test_think_filter_hides_thoughts_split_across_chunks():
    assert feed_all(["<thi", "nk>secret ", "stuff</th", "ink>", "Answer", " here"]) == "Answer here"


def test_think_filter_passes_plain_text_through():
    assert feed_all(["Hello", " world"]) == "Hello world"


def test_think_filter_text_that_merely_starts_with_less_than():
    assert feed_all(["<b", "> bold"]) == "<b> bold"


def test_embedding_models_are_not_chat_models():
    assert not is_chat_model("sentence-transformer-mini")
    assert not is_chat_model("nomic-embed-text")
    assert is_chat_model("Jan-v3.5-4B-Q4_K_XL")
