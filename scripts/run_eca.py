"""
CLI entry point for the ECA pipeline.

Usage:
    # With prompt as argument:
    python scripts/run_eca.py "My ML model overfits. How do I fix it?"

    # Interactive mode (prompts for input):
    python scripts/run_eca.py

    # With options:
    python scripts/run_eca.py --k-gaps 5 --model anthropic/claude-sonnet-4-5 "prompt here"

    # Quiet mode (response only, no audit):
    python scripts/run_eca.py --quiet "prompt here"

    # JSON output:
    python scripts/run_eca.py --json "prompt here"

Installed as a script via pyproject.toml:
    run-eca "prompt here"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run-eca",
        description="Run the Epistemic Coverage Architecture on a prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  run-eca "My database queries are slow. What approaches should I consider?"
  run-eca --k-gaps 5 "How do I reduce employee turnover?"
  run-eca --quiet "Why does my startup grow slowly?"
  run-eca --json "How do I stop my ML model from overfitting?" | python -m json.tool
        """,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The problem or question to analyze. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--k-gaps",
        type=int,
        default=None,
        metavar="N",
        help="Number of coverage gap domains to explore (default: 3 from config).",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="OpenRouter model ID (default: anthropic/claude-sonnet-4-5).",
    )
    parser.add_argument(
        "--dtg",
        default=None,
        metavar="PATH",
        help="Path to DTG JSON file (default: data/dtg_300.json).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the synthesized response, no audit output.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output results as JSON (response + audit).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored terminal output.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    from efa import ECA
    from efa.config import DEFAULT_MODEL, DTG_PATH, K_GAPS

    # Resolve prompt
    if args.prompt:
        prompt = args.prompt.strip()
    else:
        if sys.stdin.isatty():
            print("Enter your problem or question (Ctrl-D when done):")
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("Error: no prompt provided.", file=sys.stderr)
        sys.exit(1)

    dtg_path = Path(args.dtg) if args.dtg else DTG_PATH
    if not dtg_path.exists():
        print(
            f"Error: DTG not found at {dtg_path}\n"
            f"Run `build-dtg` (or `python scripts/build_dtg.py`) first.",
            file=sys.stderr,
        )
        sys.exit(1)

    k_gaps = args.k_gaps or K_GAPS
    model = args.model or DEFAULT_MODEL

    if not args.quiet and not args.json_output:
        _print_header(f"ECA Pipeline  ·  k_gaps={k_gaps}  ·  {model}", args.no_color)
        print(f"\nPrompt: {prompt}\n")

    eca = ECA(dtg_path=dtg_path, model=model, k_gaps=k_gaps)
    result = eca.run(prompt)

    if args.json_output:
        output = {
            "prompt": prompt,
            "response": result.response,
            "coverage_audit": result.coverage_audit,
            "outside_frame_concepts": [
                {
                    "concept": dc.concept,
                    "source_domain": dc.source_domain,
                    "score": round(dc.score, 4),
                    "distance_from_frame": round(dc.distance_from_c0, 4),
                }
                for dc in result.concepts
            ],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if args.quiet:
        print(result.response)
        return

    # Full output with audit
    _print_section("Coverage Audit", args.no_color)
    print(result.summary())

    _print_section("Outside-Frame Concepts Found", args.no_color)
    if result.concepts:
        for dc in result.concepts:
            print(f"  [{dc.source_domain}] {dc.concept}  (score={dc.score:.3f})")
    else:
        print("  (none surfaced — frame may already cover relevant domains)")

    _print_section("Synthesized Response", args.no_color)
    print(result.response)

    if not args.quiet:
        remaining = result.coverage_audit.get("remaining_unexplored", [])
        if remaining:
            _print_section(f"Known Remaining Gaps (not explored)", args.no_color)
            print(f"  {', '.join(remaining)}")


def _print_header(text: str, no_color: bool) -> None:
    line = "=" * 60
    if no_color:
        print(f"\n{line}\n{text}\n{line}")
    else:
        print(f"\n\033[1;36m{line}\n{text}\n{line}\033[0m")


def _print_section(title: str, no_color: bool) -> None:
    if no_color:
        print(f"\n--- {title} ---")
    else:
        print(f"\n\033[1;33m--- {title} ---\033[0m")


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
