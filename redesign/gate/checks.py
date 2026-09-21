"""Correctness checks, in three tiers. Register item Z5.

Why this is not a benchmark
---------------------------
Quantised KV does not crash. It produces fluent, confident, wrong output --
K3-DEPLOYMENT.md 7.3's silent-garbage failure -- which every throughput
benchmark reports as a success. So no check here measures speed, and none
compares token-exact output across profiles: kernel and layout changes move
numerics legitimately, and a token-exact comparison would fail on correct
builds while passing on a subtly broken one.

Instead:

  Tier 1  corruption      self-consistency WITHIN a run. Greedy decoding must
                          be reproducible; degenerate repetition and control
                          characters are hard failures.
  Tier 2  reasoning       task success on extractable answers. Robust to
                          numerics, sensitive to semantic drift.
  Tier 3  long context    needle retrieval at controlled depths. This is where
                          quantised KV degrades first and worst, and the only
                          tier that exercises the regime fp8 actually changes.

Tier 3 is the one that matters for A2. A build can pass tiers 1 and 2 with a
badly quantised KV cache and still lose the middle of a 200k prompt.
"""

from __future__ import annotations

import abc
import json
import re
import string
from dataclasses import dataclass, field

from .client import Completion, CompletionError, GateClient


def _longest_adjacent_run(words: list[str]) -> int:
    run_length, longest, previous = 0, 0, None
    for word in words:
        run_length = run_length + 1 if word == previous else 1
        longest = max(longest, run_length)
        previous = word
    return longest


def _longest_phrase_run(words: list[str]) -> tuple[int, int]:
    best_run, best_n = 1, 1
    for n in PHRASE_NGRAM_SIZES:
        if len(words) < n * 2:
            continue
        run, longest = 1, 1
        prev = tuple(words[0:n])
        i = n
        while i + n <= len(words):
            cur = tuple(words[i:i + n])
            if cur == prev:
                run += 1
                longest = max(longest, run)
                i += n
            else:
                run = 1
                prev = cur
                i += 1
        if longest > best_run:
            best_run, best_n = longest, n
    return best_run, best_n

TIER_CORRUPTION = 1
TIER_REASONING = 2
TIER_LONG_CONTEXT = 3

# A greedy run repeated this many times must agree with itself every time.
DETERMINISM_SAMPLES = 3

# A token repeated more than this consecutively is degenerate, not fluent.
MAX_CONSECUTIVE_REPEATS = 12

# A phrase (2-8 words) looping this many times is degenerate, not fluent.
MAX_PHRASE_REPEATS = 8
PHRASE_NGRAM_SIZES = range(2, 9)

# Depths as a fraction through the filler, chosen to include both edges and the
# middle -- the middle is where a lossy KV cache loses information first.
NEEDLE_DEPTHS = (0.1, 0.5, 0.9)

# English filler averages ~6.7 characters per token, not 4.
FILLER_CHARS_PER_TOKEN = 6.7

FILLER_SENTENCES = (
    "The archive catalogues routine maintenance records. ",
    "Bay three was inspected on Tuesday and needed no follow-up. ",
    "The east wing stores surplus fasteners and spare gaskets. ",
    "Night shift logged a humidity spike in corridor B and reset the sensor. ",
    "A crate of unmarked washers sat beside the west loading dock. ",
    "Calendar note: the freight elevator is reserved for parts, not people. ",
    "The duty engineer initialled the clipboard and moved to the next bay. ",
    "Rain against the north glass made the afternoon inventory run long. ",
)

CLEAN_TEXT_EXPECTED = "ready"
EXACT_ANSWER_MAX_LAST_LINE = 48


@dataclass(frozen=True)
class CheckResult:
    name: str
    tier: int
    passed: bool
    detail: str
    latency_seconds: float = 0.0
    observed: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


class Check(abc.ABC):
    tier: int = TIER_CORRUPTION

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def run(self, client: GateClient) -> CheckResult: ...

    def _fail(self, detail: str, **kw) -> CheckResult:
        return CheckResult(self.name, self.tier, False, detail, **kw)

    def _pass(self, detail: str, **kw) -> CheckResult:
        return CheckResult(self.name, self.tier, True, detail, **kw)


# ---------------------------------------------------------------------------
# Tier 1 -- corruption
# ---------------------------------------------------------------------------

def _extracted_integers(text: str) -> tuple[int, ...]:
    return tuple(int(match) for match in re.findall(r"-?\d+", text))


class GreedyDeterminism(Check):
    tier = TIER_CORRUPTION
    name = "greedy-determinism"

    def run(self, client: GateClient) -> CheckResult:
        prompt = "List the first eight prime numbers, comma separated, nothing else."
        outputs, elapsed = [], 0.0
        for _ in range(DETERMINISM_SAMPLES):
            try:
                # Reasoning models spend the first tokens on hidden thinking;
                # 64 was finishing mid-list and comparing truncated prefixes.
                completion = client.complete(prompt, max_tokens=256)
            except CompletionError as exc:
                return self._fail(f"request failed: {exc}")
            outputs.append(completion.text.strip())
            elapsed += completion.latency_seconds

        extracted = [_extracted_integers(text) for text in outputs]
        if len(set(extracted)) != 1 or not extracted[0]:
            return self._fail(
                f"{len(set(outputs))} distinct outputs from {DETERMINISM_SAMPLES} greedy runs",
                latency_seconds=elapsed, observed=" | ".join(sorted(set(outputs)))[:300],
            )
        return self._pass("greedy decoding is reproducible",
                          latency_seconds=elapsed, observed=outputs[0][:200])


class NoDegenerateRepetition(Check):
    tier = TIER_CORRUPTION
    name = "no-degenerate-repetition"

    def run(self, client: GateClient) -> CheckResult:
        prompt = "Write three sentences about the history of the printing press."
        try:
            completion = client.complete(prompt, max_tokens=256)
        except CompletionError as exc:
            return self._fail(f"request failed: {exc}")

        words = completion.text.split()
        longest_word = _longest_adjacent_run(words)
        longest_phrase, phrase_n = _longest_phrase_run(words)

        if longest_word > MAX_CONSECUTIVE_REPEATS:
            return self._fail(
                f"a token repeats {longest_word} times consecutively",
                latency_seconds=completion.latency_seconds,
                observed=completion.text[:300],
            )
        if longest_phrase > MAX_PHRASE_REPEATS:
            return self._fail(
                f"a {phrase_n}-word phrase repeats {longest_phrase} times",
                latency_seconds=completion.latency_seconds,
                observed=completion.text[:300],
            )
        return self._pass(
            f"longest word run {longest_word}, longest phrase run {longest_phrase}",
            latency_seconds=completion.latency_seconds,
        )


class CleanText(Check):
    tier = TIER_CORRUPTION
    name = "clean-text"

    def run(self, client: GateClient) -> CheckResult:
        prompt = "Reply with exactly: ready"
        try:
            completion = client.complete(prompt, max_tokens=32)
        except CompletionError as exc:
            return self._fail(f"request failed: {exc}")

        text = completion.text
        if not text.strip():
            return self._fail("empty completion",
                              latency_seconds=completion.latency_seconds)

        if CLEAN_TEXT_EXPECTED not in text.casefold():
            return self._fail(
                f"expected {CLEAN_TEXT_EXPECTED!r} in the completion",
                latency_seconds=completion.latency_seconds, observed=text[:200],
            )

        allowed = set(string.printable)
        bad = {c for c in text if c not in allowed and not c.isprintable()}
        if bad:
            return self._fail(
                f"{len(bad)} non-printable character(s) in output",
                latency_seconds=completion.latency_seconds, observed=repr(text[:200]),
            )
        return self._pass("output contains the requested reply and is printable",
                          latency_seconds=completion.latency_seconds)


# ---------------------------------------------------------------------------
# Tier 2 -- reasoning
# ---------------------------------------------------------------------------

@dataclass
class ExactAnswer(Check):
    """A prompt with one extractable correct answer.

    Answer extraction is a regex over the whole completion rather than an
    equality test on it, so a model that reasons aloud before answering is not
    failed for being verbose.
    """

    prompt: str
    expected: str
    label: str
    pattern: str = r"[-+]?\d[\d,]*"
    tier: int = field(default=TIER_REASONING, init=False)

    @property
    def name(self) -> str:
        return f"reasoning-{self.label}"

    def run(self, client: GateClient) -> CheckResult:
        try:
            completion = client.complete(self.prompt, max_tokens=512)
        except CompletionError as exc:
            return self._fail(f"request failed: {exc}")

        lines = [ln.strip() for ln in completion.text.strip().splitlines() if ln.strip()]
        if not lines:
            return self._fail("empty completion",
                              latency_seconds=completion.latency_seconds)
        last = lines[-1]
        if len(last) > EXACT_ANSWER_MAX_LAST_LINE:
            return self._fail(
                f"last line is {len(last)} chars of prose, not an extractable answer",
                latency_seconds=completion.latency_seconds, observed=last[:200],
            )
        found = re.findall(self.pattern, last)
        normalised = [f.replace(",", "").strip().lower() for f in found]
        if self.expected.lower() in normalised:
            return self._pass(f"found {self.expected} on the last line",
                              latency_seconds=completion.latency_seconds)
        return self._fail(
            f"expected {self.expected} on the last line, extracted {normalised or 'nothing'}",
            latency_seconds=completion.latency_seconds, observed=last[:300],
        )


class StructuredOutput(Check):
    tier = TIER_REASONING
    name = "reasoning-structured-output"

    def run(self, client: GateClient) -> CheckResult:
        prompt = (
            'Reply with only a JSON object, no prose and no code fence, with keys '
            '"city" set to "Delhi" and "count" set to the number 3.'
        )
        try:
            completion = client.complete(prompt, max_tokens=128)
        except CompletionError as exc:
            return self._fail(f"request failed: {exc}")

        text = completion.text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return self._fail("no JSON object in output",
                              latency_seconds=completion.latency_seconds,
                              observed=text[:300])
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return self._fail(f"invalid JSON: {exc}",
                              latency_seconds=completion.latency_seconds,
                              observed=match.group(0)[:300])

        if parsed.get("city") != "Delhi" or parsed.get("count") != 3:
            return self._fail(f"wrong values: {parsed}",
                              latency_seconds=completion.latency_seconds)
        return self._pass("valid JSON with the requested values",
                          latency_seconds=completion.latency_seconds)


# ---------------------------------------------------------------------------
# Tier 3 -- long context
# ---------------------------------------------------------------------------

@dataclass
class NeedleRetrieval(Check):
    """Hides a fact at a controlled depth in a long prompt and asks for it back.

    The single most sensitive check for quantised KV: a lossy cache degrades in
    the middle of a long context long before it produces anything that looks
    broken at short context.
    """

    context_tokens: int
    depth: float
    tier: int = field(default=TIER_LONG_CONTEXT, init=False)

    NEEDLE = "The maintenance code for bay seventeen is {code}."
    CODE = "PHOENIX-4471"

    @property
    def name(self) -> str:
        return f"needle-{self.context_tokens // 1000}k-depth{int(self.depth * 100)}"

    def _filler(self, index: int) -> str:
        return FILLER_SENTENCES[index % len(FILLER_SENTENCES)]

    def estimated_prompt_tokens(self) -> int:
        return max(1, int(len(self._build_prompt()) / FILLER_CHARS_PER_TOKEN))

    def _build_prompt(self) -> str:
        target_chars = int(self.context_tokens * FILLER_CHARS_PER_TOKEN)
        mean_len = sum(len(s) for s in FILLER_SENTENCES) / len(FILLER_SENTENCES)
        filler_units = max(1, int(target_chars / mean_len))
        before = int(filler_units * self.depth)
        haystack = "".join(self._filler(i) for i in range(before))
        haystack += self.NEEDLE.format(code=self.CODE) + " "
        haystack += "".join(self._filler(i) for i in range(before, filler_units))
        return (
            "Read the archive below and answer the question that follows.\n\n"
            + haystack
            + "\n\nQuestion: what is the maintenance code for bay seventeen? "
              "Reply with the code only."
        )

    def run(self, client: GateClient) -> CheckResult:
        try:
            completion = client.complete(self._build_prompt(), max_tokens=64)
        except CompletionError as exc:
            return self._fail(f"request failed: {exc}")

        if self.CODE.lower() in completion.text.lower():
            return self._pass(
                f"retrieved at depth {self.depth:.0%} of ~{self.context_tokens:,} tokens",
                latency_seconds=completion.latency_seconds,
            )
        return self._fail(
            f"needle not retrieved at depth {self.depth:.0%} "
            f"of ~{self.context_tokens:,} tokens",
            latency_seconds=completion.latency_seconds,
            observed=completion.text[:200],
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_registry(long_context_tokens: tuple[int, ...] = (32_000, 128_000)) -> list[Check]:
    checks: list[Check] = [
        GreedyDeterminism(),
        NoDegenerateRepetition(),
        CleanText(),
        ExactAnswer(
            prompt="A train travels 60 km in the first hour, 80 km in the second, "
                   "and 100 km in the third. How many kilometres in total? "
                   "End your reply with the number alone.",
            expected="240", label="arithmetic",
        ),
        ExactAnswer(
            prompt="If all Blips are Trids, and all Trids are Grons, and there are "
                   "12 Blips, at least how many Grons are there? "
                   "End your reply with the number alone.",
            expected="12", label="syllogism",
        ),
        StructuredOutput(),
    ]
    for tokens in long_context_tokens:
        for depth in NEEDLE_DEPTHS:
            checks.append(NeedleRetrieval(context_tokens=tokens, depth=depth))
    return checks
