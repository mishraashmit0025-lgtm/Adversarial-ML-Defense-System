"""CLI: python -m adv_defense [--data path/to/MachineLearningCVE] [--eps 0.5]"""
import argparse

from .pipeline import run, to_markdown


def main(argv=None):
    ap = argparse.ArgumentParser(prog="adv_defense")
    ap.add_argument("--data", help="CICIDS 2017 CSV file or directory; omit to use synthetic flows")
    ap.add_argument("--n", type=int, default=40000, help="synthetic flows to generate")
    ap.add_argument("--max-rows-per-file", type=int, default=50000)
    ap.add_argument("--eps", type=float, default=1.0, help="max relative increase of each mutable feature")
    ap.add_argument("--n-attack", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-false-alarm", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    r = run(a.data, a.n, a.eps, a.n_attack, a.seed, a.out, a.max_rows_per_file, a.epochs, a.max_false_alarm)
    print(to_markdown(r))
    print(f"(finished in {r['seconds']}s; report written to {a.out}/report.md)")


if __name__ == "__main__":
    main()
