"""python -m research_agent "your claim" [--rows N] [--json]"""
import argparse
import json

from .pipeline import check_claim


def main():
    parser = argparse.ArgumentParser(description="Check a claim against arXiv papers.")
    parser.add_argument("claim")
    parser.add_argument("--rows", type=int, default=5, help="papers to fetch from arXiv (default 5)")
    parser.add_argument("--json", action="store_true", help="print the raw result as JSON")
    args = parser.parse_args()

    out = check_claim(args.claim, rows=args.rows)
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"\nClaim: {args.claim}\n")
    for r in out["results"]:
        print(f"[{r['verdict'].upper()}] {r['title']} ({r['year']})")
        print(f"  {r['url']}")
        if r["quote"]:
            print(f'  "{r["quote"]}"')
    d = out["diagnostics"]
    print(
        f"\n{d['sources_found']} found, {d['kept_after_prefilter']} kept, {d['judged']} judged, "
        f"{d['pdf_failures']} PDF failures, {d['guard_downgrades']} quotes rejected by the guard"
    )


if __name__ == "__main__":
    main()
