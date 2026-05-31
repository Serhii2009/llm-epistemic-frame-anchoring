"""
compare-eca: side-by-side comparison of ECA vs vanilla LLM on the same prompt.

This is the empirical test for whether ECA is doing something distinctive.

Usage:
    compare-eca "Why do school reform initiatives consistently fail?"
    compare-eca --runs 3 "same prompt"    # show vanilla variance across N runs
    compare-eca --json "prompt"           # JSON output for programmatic use
    compare-eca --quiet "prompt"          # delta only (what ECA added)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SEP = "═" * 64


def _hr(title: str = "", color: str = "") -> None:
    if title:
        pad = (64 - len(title) - 2) // 2
        line = f"{'═' * pad} {title} {'═' * (64 - pad - len(title) - 2)}"
    else:
        line = SEP
    if color:
        print(f"\033[{color}m{line}\033[0m")
    else:
        print(line)


def _section(title: str, no_color: bool = False) -> None:
    if no_color:
        print(f"\n{SEP}\n{title}\n{SEP}")
    else:
        print(f"\n\033[1;36m{SEP}\n{title}\n{SEP}\033[0m")


def run_comparison(
    prompt: str,
    vanilla_runs: int = 1,
    json_output: bool = False,
    quiet: bool = False,
    no_color: bool = False,
    k_gaps: int = 3,
    model: str | None = None,
) -> None:
    from efa import ECA
    from efa.config import DEFAULT_MODEL, DTG_PATH
    from efa.embeddings import embed
    from efa.llm import LLMClient
    from efa.pipeline.delta import ConceptDeltaExtractor
    from efa.pipeline.verify import CoverageVerifier

    use_model = model or DEFAULT_MODEL
    llm = LLMClient(model=use_model)
    extractor = ConceptDeltaExtractor(llm)
    verifier = CoverageVerifier()

    if not DTG_PATH.exists():
        print(f"ERROR: DTG not found at {DTG_PATH}. Run `build-dtg` first.", file=sys.stderr)
        sys.exit(1)

    if not quiet and not json_output:
        print(f"\nPrompt: {prompt}")
        print(f"Model:  {use_model}  |  k_gaps={k_gaps}  |  vanilla_runs={vanilla_runs}\n")

    # ── Vanilla runs ──────────────────────────────────────────────────────────
    vanilla_responses: list[str] = []
    for i in range(vanilla_runs):
        temp = 0.9 if vanilla_runs > 1 else 0.7
        resp = llm.complete(prompt, temperature=temp)
        vanilla_responses.append(resp)

    primary_vanilla = vanilla_responses[0]

    # ── ECA run ───────────────────────────────────────────────────────────────
    eca = ECA(dtg_path=DTG_PATH, model=use_model, k_gaps=k_gaps)
    result = eca.run(prompt)

    # ── Delta: ECA concepts not in vanilla response ───────────────────────────
    # Extract noun phrases from vanilla response for comparison
    vanilla_phrases = extractor._extract_noun_phrases(primary_vanilla)

    # For each ECA concept, check if it appears in the vanilla response text
    # using cosine similarity (same logic as CoverageVerifier)
    eca_only: list[dict] = []
    also_in_vanilla: list[dict] = []

    if result.concepts:
        vanilla_phrase_vecs = [embed.encode(p) for p in vanilla_phrases] if vanilla_phrases else []

        for dc in result.concepts:
            concept_vec = embed.encode(dc.concept)
            found_in_vanilla = False

            # Exact substring check
            if any(dc.concept.lower() in p.lower() or p.lower() in dc.concept.lower()
                   for p in vanilla_phrases):
                found_in_vanilla = True

            # Cosine similarity check (threshold 0.75)
            if not found_in_vanilla and vanilla_phrase_vecs:
                import numpy as np
                sims = [float(concept_vec @ v) for v in vanilla_phrase_vecs]
                if max(sims) >= 0.75:
                    found_in_vanilla = True

            # Also check raw vanilla response text
            if not found_in_vanilla and dc.concept.lower() in primary_vanilla.lower():
                found_in_vanilla = True

            entry = {
                "concept": dc.concept,
                "source_domain": dc.source_domain,
                "score": round(dc.score, 4),
                "in_vanilla": found_in_vanilla,
            }
            (also_in_vanilla if found_in_vanilla else eca_only).append(entry)

    # ── Vanilla variance analysis (if multiple runs) ──────────────────────────
    concept_run_counts: dict[str, int] = {}
    if vanilla_runs > 1:
        all_vanilla_phrases: list[set[str]] = []
        for vresp in vanilla_responses:
            phrases = set(extractor._extract_noun_phrases(vresp))
            all_vanilla_phrases.append(phrases)

        all_concepts = set()
        for phrases in all_vanilla_phrases:
            all_concepts.update(phrases)

        for concept in sorted(all_concepts):
            count = sum(1 for ps in all_vanilla_phrases if concept in ps)
            if count < vanilla_runs:  # only show concepts that don't appear in every run
                concept_run_counts[concept] = count

    # ── Output ────────────────────────────────────────────────────────────────
    if json_output:
        out = {
            "prompt": prompt,
            "model": use_model,
            "vanilla_response": primary_vanilla,
            "eca_response": result.response,
            "coverage_audit": result.coverage_audit,
            "eca_only_concepts": eca_only,
            "concepts_also_in_vanilla": also_in_vanilla,
            "vanilla_variance": concept_run_counts if vanilla_runs > 1 else None,
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if quiet:
        # Delta only
        if eca_only:
            print("ECA surfaced — not in vanilla:")
            for e in eca_only:
                print(f"  [{e['source_domain']}] {e['concept']}")
        else:
            print("(no concepts surfaced by ECA that weren't also in the vanilla response)")
        return

    # Full output
    _section("VANILLA LLM", no_color)
    print(primary_vanilla)

    _section("ECA RESPONSE", no_color)
    print(result.response)

    _section("COVERAGE AUDIT", no_color)
    audit = result.coverage_audit
    print(f"  Frame:           {audit.get('frame_summary', '(none)')}")
    print(f"  Primary node:    {audit.get('frame_concepts', [])[:3]}")
    print(f"  Activated:       {audit.get('activated_domains', [])}")
    print(f"  Gap domains:     {audit.get('explored_gap_domains', [])}")
    print(f"  Concepts found:  {audit.get('concepts_after_probe', 0)}")
    print(f"  Contamination:   {audit.get('contamination_warning', False)}")

    _section("DELTA: what ECA surfaced that vanilla didn't mention", no_color)
    if eca_only:
        print(f"  {len(eca_only)} concept(s) NOT found in vanilla response:\n")
        for e in eca_only:
            print(f"  [{e['source_domain']}] {e['concept']}  (score={e['score']})")
    else:
        print("  (none — vanilla LLM appears to cover the same concepts)")

    if also_in_vanilla:
        print(f"\n  {len(also_in_vanilla)} concept(s) also present in vanilla:")
        for e in also_in_vanilla[:5]:
            print(f"  [{e['source_domain']}] {e['concept']}")
        if len(also_in_vanilla) > 5:
            print(f"  ... and {len(also_in_vanilla) - 5} more")

    if vanilla_runs > 1:
        _section(f"VANILLA VARIANCE: {vanilla_runs} runs", no_color)
        # Show concepts that appear in fewer than all runs, sorted by frequency ascending
        unstable = [(c, n) for c, n in concept_run_counts.items() if n == 0 or n < vanilla_runs]
        unstable.sort(key=lambda x: x[1])
        never = [(c, n) for c, n in unstable if n == 0]
        sometimes = [(c, n) for c, n in unstable if 0 < n < vanilla_runs]

        print(f"\n  Concepts in ALL {vanilla_runs} runs: "
              f"{sum(1 for n in concept_run_counts.values() if n == vanilla_runs)} concepts "
              f"(consistent)")

        if never[:10]:
            print(f"\n  Never surfaced by vanilla (0/{vanilla_runs} runs) — EFA candidates:")
            for c, _ in never[:10]:
                print(f"    '{c}'")

        if sometimes[:10]:
            print(f"\n  Inconsistently surfaced by vanilla (1-{vanilla_runs-1}/{vanilla_runs} runs):")
            for c, n in sometimes[:10]:
                print(f"    '{c}': {n}/{vanilla_runs} runs")

    # ── Verdict ───────────────────────────────────────────────────────────────
    _section("VERDICT", no_color)
    if eca_only:
        print(f"  ECA produced {len(eca_only)} concept(s) not in the vanilla response.")
        print(f"  These are the candidates for genuine ECA value — inspect them manually:")
        for e in eca_only:
            print(f"  → [{e['source_domain']}] {e['concept']}")
    else:
        print("  ECA produced NO concepts that weren't already present in the vanilla response.")
        print("  This suggests ECA is not adding value for this prompt, OR the gap domain")
        print("  selection is still miscalibrated (check COVERAGE AUDIT above).")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compare-eca",
        description="Compare ECA vs vanilla LLM side by side on the same prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  compare-eca "Why do school reform initiatives fail?"
  compare-eca --runs 3 "same prompt — shows vanilla stochasticity"
  compare-eca --quiet "prompt" | less
  compare-eca --json "prompt" | python -m json.tool
        """,
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to analyze.")
    parser.add_argument("--runs", type=int, default=1, metavar="N",
                        help="Number of vanilla LLM runs (default: 1). Use 3+ to show variance.")
    parser.add_argument("--k-gaps", type=int, default=3, metavar="N",
                        help="ECA coverage gap domains to explore (default: 3).")
    parser.add_argument("--model", default=None, help="OpenRouter model ID.")
    parser.add_argument("--json", dest="json_output", action="store_true",
                        help="Output results as JSON.")
    parser.add_argument("--quiet", action="store_true",
                        help="Print only the delta (what ECA surfaced that vanilla didn't).")
    parser.add_argument("--no-color", action="store_true", help="Disable color output.")
    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt.strip()
    else:
        if sys.stdin.isatty():
            print("Enter your prompt (Ctrl-D when done):")
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("Error: no prompt provided.", file=sys.stderr)
        sys.exit(1)

    try:
        run_comparison(
            prompt=prompt,
            vanilla_runs=args.runs,
            json_output=args.json_output,
            quiet=args.quiet,
            no_color=args.no_color,
            k_gaps=args.k_gaps,
            model=args.model,
        )
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
