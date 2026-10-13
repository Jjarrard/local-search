"""question -> keywords -> search -> grow -> filter -> squeeze -> notes.

The design rule: code does everything it can, and the model only answers
small closed questions it's hard to get wrong. Every model answer is
checked by code, and every failure has a fallback, so a model that replies
with garbage makes the result worse, never broken.

  keywords  model lists search words      fallback: the question's own words
  search    BM25 over paragraphs          pure code
  grow      "does B continue A?" YES/NO   only asked when cheap rules can't tell
  filter    "is this relevant?" YES/NO    unparseable answer keeps the passage
  squeeze   "copy the useful sentences"   every line is matched back to a real
                                          sentence in the source, so the model
                                          can't put words in a book's mouth

The final TLDR prompt is built here but streamed by the caller.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .extract import split_sentences
from .index import Hit, Index, fts_query

# prompt, max_tokens -> reply (already stripped of <think> blocks)
Ask = Callable[[str, int], str]
Log = Callable[[str], None]

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most my
myself no nor not now of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves tell find know out give explain describe info information anything something
everything thing things please also really much many get got make made show say said think
book books text
""".split())

# A paragraph that opens like this is almost always still talking about
# whatever the previous one was - no need to spend a model call on it.
CONTINUATION_OPENERS = (
    "he ", "she ", "it ", "they ", "this ", "these ", "those ", "that ", "his ", "her ",
    "its ", "their ", "however", "but ", "and ", "also", "moreover", "furthermore",
    "in addition", "additionally", "thus", "therefore", "hence", "consequently",
    "for example", "for instance", "such ", "similarly", "likewise", "meanwhile",
    "as a result", "in fact", "indeed", "nevertheless", "still,", "yet ",
)
HEADING = re.compile(r"^(chapter|part|section|book|volume|appendix|contents|index)\b", re.I)

MAX_PASSAGE_CHARS = 3000   # stop growing past this; a tiny model can't read more
JUDGE_CHARS = 1500         # how much of a passage the YES/NO filter sees
SQUEEZE_CHARS = 2400       # how much of a passage the squeeze step sees
TLDR_REPLY_TOKENS = 400
MIN_HIT_CHARS = 100        # shorter hits are titles, headings, index entries


@dataclass
class Passage:
    file_id: int
    path: str
    start: int                  # first paragraph number (0-based)
    end: int                    # last paragraph number (inclusive)
    paragraphs: list[str]
    score: float                # best bm25 of the hits inside it (lower = better)
    extract: list[str] = field(default_factory=list)  # the squeezed sentences

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def label(self) -> str:
        span = f"¶{self.start + 1}" if self.start == self.end else f"¶{self.start + 1}-{self.end + 1}"
        return f"{self.path} {span}"


@dataclass
class Gathered:
    question: str
    keywords: list[str]
    passages: list[Passage]
    searched: int = 0           # raw search hits looked at


# --- small pure helpers ----------------------------------------------------


def content_words(text: str) -> list[str]:
    words = re.findall(r"[\w'-]+", text.lower())
    out = [w.strip("'-") for w in words]
    return list(dict.fromkeys(w for w in out if len(w) > 1 and w not in STOPWORDS))


def question_terms(question: str) -> list[str]:
    """Quoted phrases stay whole; everything else becomes content words."""
    phrases = [p.strip().lower() for p in re.findall(r'"([^"]+)"', question) if p.strip()]
    rest = re.sub(r'"[^"]*"', " ", question)
    return list(dict.fromkeys(phrases + content_words(rest)))


def parse_keywords(reply: str, limit: int = 6) -> list[str]:
    """Pull a keyword list out of whatever the model said.

    Tolerates numbering, bullets, a "Keywords:" preamble, quotes, and the
    model rambling - anything longer than a few words is not a keyword.
    """
    reply = re.sub(r"^\s*keywords?\s*:", "", reply.strip(), flags=re.I)
    out = []
    for raw in re.split(r"[,\n;]", reply):
        kw = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", raw).strip().strip("\"'.`").lower()
        if not kw or len(kw.split()) > 3 or len(kw) > 40:
            continue
        if not re.search(r"\w", kw) or kw in STOPWORDS:
            continue
        out.append(kw)
    return list(dict.fromkeys(out))[:limit]


def parse_yes_no(reply: str) -> bool | None:
    """YES -> True, NO -> False, anything else -> None (caller decides)."""
    m = re.search(r"\b(yes|no)\b", reply.lower())
    if not m:
        return None
    return m.group(1) == "yes"


def _roots(terms: list[str]) -> list[str]:
    """Crude stems for cheap matching: 'volcanoes' -> 'volcano'."""
    roots = []
    for term in terms:
        for w in re.findall(r"\w+", term.lower()):
            if w in STOPWORDS:
                continue
            roots.append(w[: max(4, len(w) - 2)] if len(w) > 4 else w)
    return list(dict.fromkeys(roots))


def mentions(text: str, roots: list[str]) -> int:
    """How many distinct roots appear at the start of some word in text."""
    words = set(re.findall(r"\w+", text.lower()))
    return sum(1 for r in roots if any(w.startswith(r) for w in words))


def looks_like_heading(para: str) -> bool:
    p = para.strip()
    if HEADING.match(p) and len(p) < 120:
        return True
    return len(p) < 60 and (p.isupper() or not re.search(r"[.!?:;,]$", p))


def looks_like_junk(para: str) -> bool:
    """Tables of contents, back-of-book indexes, ascii tables: lots of search
    words, no actual sentences. Mostly CAPITALS, or mostly numbers.

    Spreadsheet rows ("Month: Jan; Revenue: 1200.") are numbers-heavy on
    purpose; their "Header: value" pairs mark them as data, not junk."""
    if len(re.findall(r"\w: \S", para)) >= 2:
        return False
    letters = [c for c in para if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        return True
    tokens = para.split()
    numeric = sum(1 for t in tokens if re.fullmatch(r"[\divxlcIVXLC.,;:()-]+", t) and re.search(r"\d", t))
    return bool(tokens) and numeric / len(tokens) > 0.2


def continues_cheaply(nxt: str, roots: list[str]) -> bool | None:
    """True/False when the rules are sure, None when only a model can tell."""
    if looks_like_heading(nxt):
        return False
    if mentions(nxt, roots):
        return True
    if nxt.lower().startswith(CONTINUATION_OPENERS):
        return True
    return None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def verify_extract(reply: str, source: str) -> list[str]:
    """Map each line the model 'copied' back onto a real source sentence.

    Small models paraphrase, merge, and invent. Anything that can't be matched
    to an actual sentence (by word overlap) is thrown away, and what survives
    is the source's own wording - never the model's.
    """
    sentences = split_sentences(source)
    norm_sents = [set(_norm(s).split()) for s in sentences]
    picked: set[int] = set()
    for line in reply.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*•>])\s*", "", line).strip().strip("\"'")
        words = set(_norm(line).split())
        if len(words) < 4 or line.upper().startswith("NONE"):
            continue
        best, best_score = -1, 0.0
        for i, sw in enumerate(norm_sents):
            if not sw:
                continue
            score = len(words & sw) / len(words | sw)
            if score > best_score:
                best, best_score = i, score
        if best >= 0 and best_score >= 0.5:
            picked.add(best)
    return [sentences[i] for i in sorted(picked)]


def keyword_sentences(source: str, roots: list[str], limit: int = 3) -> list[str]:
    """No-model squeeze: the sentences that mention the most search terms."""
    sentences = split_sentences(source)
    scored = [(mentions(s, roots), i) for i, s in enumerate(sentences)]
    best = sorted((x for x in scored if x[0] > 0), key=lambda x: (-x[0], x[1]))[:limit]
    return [sentences[i] for _, i in sorted(best, key=lambda x: x[1])]


def merge_passages(passages: list[Passage]) -> list[Passage]:
    """Join passages in the same file that overlap or touch."""
    by_pos = sorted(passages, key=lambda p: (p.path, p.start))
    out: list[Passage] = []
    for p in by_pos:
        last = out[-1] if out else None
        if last and last.file_id == p.file_id and p.start <= last.end + 1:
            extra = p.end - last.end
            if extra > 0:
                last.paragraphs.extend(p.paragraphs[-extra:])
                last.end = p.end
            last.score = min(last.score, p.score)
        else:
            out.append(Passage(p.file_id, p.path, p.start, p.end, list(p.paragraphs), p.score))
    return sorted(out, key=lambda p: p.score)


# --- prompts ---------------------------------------------------------------
# Short, one job each, and they end where the answer should begin.

def keywords_prompt(question: str) -> str:
    return (
        "List 3 to 6 search keywords for finding the answer to this question in a book.\n"
        "Reply with only the keywords, separated by commas.\n\n"
        f"Question: {question}\n\nKeywords:"
    )


def continues_prompt(question: str, a: str, b: str) -> str:
    return (
        f"Topic: {question}\n\n"
        f"Paragraph A:\n{a[-700:]}\n\n"
        f"Paragraph B:\n{b[:700]}\n\n"
        "Does paragraph B continue talking about the topic? Answer YES or NO.\nAnswer:"
    )


def relevant_prompt(question: str, text: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Text:\n{text[:JUDGE_CHARS]}\n\n"
        "Does the text contain information that helps answer the question? Answer YES or NO.\nAnswer:"
    )


def squeeze_prompt(question: str, text: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Text:\n{text[:SQUEEZE_CHARS]}\n\n"
        "Copy the sentences from the text that help answer the question.\n"
        "Copy them word for word, one per line. If none help, write NONE."
    )


def tldr_prompt(question: str, passages: list[Passage]) -> str:
    notes = "\n".join(f"[{i}] {' '.join(p.extract)}" for i, p in enumerate(passages, 1))
    return (
        f"Question: {question}\n\n"
        f"Notes:\n{notes}\n\n"
        "Using only the notes above, write a short answer (TLDR) to the question in 3 to 5 sentences. "
        "After each fact, put the note number in brackets, like [1]. "
        "If the notes do not answer the question, say so."
    )


# --- the pipeline ----------------------------------------------------------


@dataclass
class Searcher:
    index: Index
    ask: Ask | None = None      # None = no model at all: search + extracts only
    top: int = 10
    max_grow: int = 3
    context_size: int = 4096
    log: Log = lambda _msg: None
    verbose: Log = lambda _msg: None
    calls: int = 0

    def _ask(self, prompt: str, max_tokens: int) -> str:
        if self.ask is None:
            return ""
        self.calls += 1
        try:
            return self.ask(prompt, max_tokens)
        except Exception as exc:  # noqa: BLE001 - a failed call is just "no answer"
            self.verbose(f"model call failed: {exc}")
            return ""

    def _yes_no(self, prompt: str) -> bool | None:
        # Room for a stray "<think></think>" some templates emit before the answer.
        return parse_yes_no(self._ask(prompt, 12))

    def gather(self, question: str) -> Gathered:
        self.calls = 0
        terms = question_terms(question) or content_words(question) or [question]

        # 1. keywords
        keywords = list(terms)
        if self.ask:
            extra = parse_keywords(self._ask(keywords_prompt(question), 48))
            keywords = list(dict.fromkeys(terms + extra))[:8]
        self.log(f"keywords: {', '.join(keywords)}")
        roots = _roots(keywords)
        # The model's keywords are good for casting a wide search net but too
        # broad to judge "is this paragraph still on topic" - only the
        # question's own words get to auto-continue a passage.
        topic_roots = _roots(terms)

        # 2. search: paragraphs with *all* the question's words first, then
        #    top up with anything matching any keyword.
        hits: list[Hit] = []
        seen: set[tuple[int, int]] = set()
        queries = [fts_query(keywords, "OR")]
        if len(terms) > 1:
            queries.insert(0, fts_query(terms, "AND"))
        for query in queries:
            for h in self.index.search(query, self.top * 3, MIN_HIT_CHARS):
                if len(hits) >= self.top:
                    break
                if (h.file_id, h.n) in seen or looks_like_junk(h.text):
                    continue
                hits.append(h)
                seen.add((h.file_id, h.n))
        self.log(f"search: {len(hits)} paragraph hit{'s' if len(hits) != 1 else ''}")
        if not hits:
            return Gathered(question, keywords, [], 0)

        passages = merge_passages(
            [Passage(h.file_id, h.path, h.n, h.n, [h.text], h.score) for h in hits]
        )

        # 3. grow each passage while the following paragraphs stay on topic
        for p in passages:
            self._grow(p, question, topic_roots)
        passages = merge_passages(passages)
        self.log(f"grow: {len(passages)} passage{'s' if len(passages) != 1 else ''}, "
                 f"{sum(len(p.paragraphs) for p in passages)} paragraphs")

        # 4. filter: drop what the model says is off-topic
        if self.ask:
            kept = []
            for p in passages:
                verdict = self._yes_no(relevant_prompt(question, p.text))
                self.verbose(f"relevant? {p.label}: {_verdict(verdict)}")
                if verdict is not False:  # unparseable -> keep; losing a real answer is worse
                    kept.append(p)
            self.log(f"filter: kept {len(kept)}/{len(passages)}")
            passages = kept

        # 5. squeeze each passage down to the sentences that matter
        squeezed = []
        for p in passages:
            sentences: list[str] = []
            if self.ask:
                sentences = verify_extract(self._ask(squeeze_prompt(question, p.text), 300), p.text)
            if not sentences:
                sentences = keyword_sentences(p.text, roots)
            self.verbose(f"squeeze {p.label}: {len(sentences)} sentence(s)")
            if sentences:
                p.extract = sentences
                squeezed.append(p)
        squeezed = self._fit_budget(question, squeezed)
        self.log(f"squeeze: {sum(len(p.extract) for p in squeezed)} sentences from {len(squeezed)} passages")
        return Gathered(question, keywords, squeezed, len(hits))

    def _grow(self, p: Passage, question: str, roots: list[str]) -> None:
        # One step back, only when the hit itself clearly continues something.
        if p.paragraphs[0].lower().startswith(CONTINUATION_OPENERS):
            prev = self.index.paragraph(p.file_id, p.start - 1)
            if prev and not looks_like_heading(prev):
                p.paragraphs.insert(0, prev)
                p.start -= 1
                self.verbose(f"grow {p.path}: took ¶{p.start + 1} (hit starts mid-thought)")
        for _ in range(self.max_grow):
            if len(p.text) > MAX_PASSAGE_CHARS:
                break
            nxt = self.index.paragraph(p.file_id, p.end + 1)
            if nxt is None:
                break
            verdict = continues_cheaply(nxt, roots)
            how = "rule"
            if verdict is None:
                verdict = self._yes_no(continues_prompt(question, p.paragraphs[-1], nxt)) if self.ask else False
                verdict = bool(verdict)  # unparseable -> stop; when unsure, take less
                how = "model"
            self.verbose(f"grow {p.path} ¶{p.end + 2}: {'continue' if verdict else 'stop'} ({how})")
            if not verdict:
                break
            p.paragraphs.append(nxt)
            p.end += 1

    def _fit_budget(self, question: str, passages: list[Passage]) -> list[Passage]:
        """Keep the best passages that fit the model's context for the TLDR."""
        budget = (self.context_size - TLDR_REPLY_TOKENS - 200) * 3  # ~3 chars per token, conservative
        budget -= len(tldr_prompt(question, []))
        out, used = [], 0
        for p in sorted(passages, key=lambda p: p.score):
            size = sum(len(s) + 1 for s in p.extract) + 8
            if out and used + size > budget:
                continue
            out.append(p)
            used += size
        return out


def _verdict(v: bool | None) -> str:
    return {True: "yes", False: "no", None: "unclear (kept)"}[v]
