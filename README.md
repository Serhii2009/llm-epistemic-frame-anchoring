# Epistemic Frame Anchoring (EFA)

This repo investigates a specific way LLMs fall short: they tend to answer from _inside_
the conceptual frame your prompt sets, and quietly under-cover genuinely relevant concepts
that live in distant fields. The project tested whether that gap is a **structural absence**
(unreachable without new architecture) or just a prompting gap. The honest finding is a
**clean negative result on the strong claim**: the distance effect is real and measurable,
but a single generic "look wider" prompt recovers most of what the naive prompt misses, so
the gap is mostly prompt-correctable, not architectural.

📄 **Full write-up (the story, the experiment, the verdict):** [`LinkedIn Article`](https://www.linkedin.com/pulse/my-first-ai-research-project-17-epistemic-frame-efa-why-kravchenko-y59sf)

> This is **research code, not a production library.** It exists to make the experiment in
> the article reproducible, and the parts that didn't work out are left in on purpose.

---

## What's here

- `efa/` — the **ECA (Epistemic Coverage Architecture)** pipeline: 8 runtime components
  (Meta-Epistemic Analyzer, Domain Topology Graph, Coverage Estimator, Contrastive Frame
  Generator, Parallel Epistemic Sampler, Concept Delta Extractor, Causal Structure Probe,
  Coverage-Aware Synthesis) plus an evaluation-only verifier (`pipeline/verify.py`).
- `efa/dtg/` + `data/dtg_300.json` — the Domain Topology Graph, the one component that
  operates _outside_ the prompt's frame.
- `bench/efa_bench.json` + `bench/SOURCES.md` — the benchmark: ~20 problems, each with
  source-cited outside-frame concepts.
- `experiments/` — the falsification study that produced the article's results
  (naive vs steelman, across Claude Sonnet and GPT-4o).
- `scripts/` — CLIs: `build-dtg`, `run-eca`, `compare-eca`, `calibrate-dtg`.

---

## Setup

Requires **Python 3.11+**.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # if not already pulled in by requirements
```

LLM calls go through **OpenRouter** (so the pipeline is model-agnostic). Put your key in a
`.env` file (see `.env.example`):

```
OPENROUTER_API_KEY=sk-or-...
```

Embeddings run **locally**, no API needed. On first use the model downloads (~1.3 GB) and
is cached.

> **Embedder note (honest):** the article refers to **BGE-M3**. The runs that produced the
> numbers below actually used **`bge-large-en-v1.5`** (see `efa/embeddings.py`), which was
> swapped in because BGE-M3 was too slow on a CPU-only laptop. The two are different models,
> so distances and exact recall numbers depend on this choice.

---

## Reproduce the result

The article's table comes from the experiment in `experiments/`. The path that produces it:

```bash
# 1. Generate responses for both conditions on both models.
#    Uses OpenRouter (real API cost — budget a few dollars; long "steelman"
#    responses dominate the bill). Use --dry for a 1-problem smoke test first.
python experiments/run_conditions.py --n 10

# 2. Score locally with the embedding proxy (free, runs on your machine).
python experiments/score_embed_only.py --tau 0.60

# 3. Recalibrate to percentile distance tiers and print the table.
python experiments/recalibrate.py
```

There is also an LLM-judge scoring path (`experiments/judge.py` → `experiments/analyze.py`),
which is the stronger detector but needs additional OpenRouter credits.

To play with the pipeline directly:

```bash
python scripts/run_eca.py "Why do my city's roads stay congested no matter what we widen?"
python scripts/compare.py "..."   # ECA output vs a plain model, side by side
```

### Reproduction target (Claude Sonnet, recall across 10 samples)

| Tier      | Naive prompt | Steelman prompt |
| --------- | ------------ | --------------- |
| Near (T1) | 1.00         | 1.00            |
| Mid (T2)  | 0.91         | 1.00            |
| Far (T3)  | 0.41         | 0.79            |

Plus: recall falls with distance under naive prompting (Spearman ρ ≈ −0.63 to −0.67 across
both models). **Exact numbers will vary** with model sampling, the embedder you use, and the
percentile cut points. The shape is the finding, not the third decimal: far concepts are
under-covered by default, and the generic steelman prompt recovers most of them.

---

## Honest notes (where the repo and the article differ)

The article is a finished narrative; the code is the ground truth. Three things worth
knowing:

1. **DTG size.** The article says "roughly 300 domains." The shipped graph
   (`data/dtg_300.json`, built from `efa/dtg/nodes.py`) has **~60 nodes**. 300 was the
   design target; v1 ships a ~60-domain taxonomy. The filename keeps the `_300` label from
   that original target.
2. **Embedder.** `bge-large-en-v1.5` in code vs BGE-M3 in the article (see the note above).
3. **Components.** "Eight components" in the article = the 8 runtime stages. There's also a
   9th, evaluation-only verifier used to score the benchmark, not part of the live pipeline.

---

## License / status

Solo research project. No warranty, no support guarantees. If you reproduce it and get
something different, that's interesting — open an issue.
