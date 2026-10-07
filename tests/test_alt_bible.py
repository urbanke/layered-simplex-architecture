"""Independent online-baseline and pinned-corpus checks for the Bible runner."""

import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gammaln, logsumexp

from lsa.alt.bible import load_corpus, sequential_classical


def literal_gt(counts):
    """Full alphabet evaluation of the manuscript definition, no update cache."""
    n = int(counts.sum())
    if n == 0:
        return np.full(len(counts), 1 / len(counts))
    prevalence = np.bincount(counts, minlength=int(counts.max()) + 2)
    raw = np.array([
        t / n if t > prevalence[t + 1]
        else (prevalence[t + 1] + 1) * (t + 1) / (n * prevalence[t])
        for t in counts
    ])
    return raw / raw.sum()


@pytest.mark.parametrize("d,n,kind", [(257, 6000, "uniform"), (31, 6000, "zipf")])
def test_incremental_gt_matches_literal_rule_through_long_stream(d, n, kind):
    rng = np.random.default_rng(823741)
    p = np.ones(d) if kind == "uniform" else 1 / np.arange(1, d + 1) ** 2
    p = p / p.sum()
    # The initial permutation forces unseen/singleton transitions and saturation;
    # the remaining stream repeatedly creates and removes occupied count classes.
    ids = np.r_[rng.permutation(d), rng.choice(d, size=n-d, p=p)]
    actual = sequential_classical(ids, d, ["good_turing_hybrid"], [-24, 0, 4])
    expected = []
    counts = np.zeros(d, dtype=int)
    for symbol in ids:
        expected.append(-math.log2(literal_gt(counts)[symbol]))
        counts[symbol] += 1
    np.testing.assert_allclose(actual["good_turing_hybrid"], expected, atol=2e-12, rtol=2e-13)


def test_online_dirichlet_mixture_chain_identity_at_multiple_prefixes():
    d = 257
    rng = np.random.default_rng(19283)
    p = rng.dirichlet(np.full(d, 0.3))
    ids = rng.choice(d, size=6000, p=p)
    exponents = list(range(-24, 5))
    methods = ["dirichlet_concentration_mixture", "add_one", "kt"]
    losses = sequential_classical(ids, d, methods, exponents)
    for n in (1, 2, 100, 1000, 6000):
        counts = np.bincount(ids[:n], minlength=d)

        def direct_evidence(a, counts=counts, n=n):
            # Evaluate the full symmetric Dirichlet formula independently;
            # the online routine only updates next-token component evidence.
            return float(gammaln(d*a) - gammaln(d*a+n)
                         + np.sum(gammaln(counts+a)-gammaln(a)))

        for method, a in (("add_one", 1.0), ("kt", 0.5)):
            assert losses[method][:n].sum() == pytest.approx(-direct_evidence(a)/math.log(2), abs=1e-7)
        logs = [direct_evidence(2.0**j) for j in exponents]
        expected_bits = -(logsumexp(logs) - math.log(len(logs))) / math.log(2)
        assert losses[methods[0]][:n].sum() == pytest.approx(expected_bits, abs=1e-7)


def _tiny_corpus(tmp_path):
    text = b"The word , the word .\n"
    compressed = gzip.compress(text, mtime=0)
    (tmp_path / "corpus.gz").write_bytes(compressed)
    record = {"compressed_sha256": hashlib.sha256(compressed).hexdigest(),
              "decompressed_sha256": hashlib.sha256(text).hexdigest(),
              "tokens": 6, "distinct_types": 5}
    (tmp_path / "corpus.json").write_text(json.dumps(record))
    return {"corpus": "corpus.gz", "manifest": "corpus.json"}, record


def test_corpus_loader_preserves_case_and_punctuation_and_checks_bytes(tmp_path):
    config, _ = _tiny_corpus(tmp_path)
    tokens, _ = load_corpus(config, tmp_path)
    assert tokens == ["The", "word", ",", "the", "word", "."]
    with (tmp_path / "corpus.gz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="compressed corpus checksum"):
        load_corpus(config, tmp_path)


@pytest.mark.parametrize("field,value,message", [
    ("decompressed_sha256", "0"*64, "uncompressed corpus checksum"),
    ("tokens", 7, "token/type counts"),
    ("distinct_types", 4, "token/type counts"),
])
def test_corpus_loader_rejects_inconsistent_content_metadata(tmp_path, field, value, message):
    config, record = _tiny_corpus(tmp_path)
    record[field] = value
    (tmp_path / "corpus.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        load_corpus(config, tmp_path)


def test_bundled_corpus_and_declared_prefix_type_counts():
    repo = Path(__file__).parents[1]
    tokens, record = load_corpus({"corpus": "data/kjv.txt.gz",
                                  "manifest": "data/kjv.manifest.json"}, repo)
    assert len(tokens) == 915_860
    assert len(set(tokens)) == 13_550
    for n, expected in record["prefix_distinct_types"].items():
        assert len(set(tokens[:int(n)])) == expected


def test_token_ids_outside_alphabet_are_rejected():
    for ids in ([0, -1], [0, 4]):
        with pytest.raises(ValueError, match="outside alphabet"):
            sequential_classical(ids, 4, ["good_turing_hybrid"], [0])
