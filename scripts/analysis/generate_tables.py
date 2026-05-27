"""
Generate LaTeX tables from collected results.
Usage: python scripts/analysis/generate_tables.py --csv all_results.csv
"""
import argparse
import pandas as pd


def generate_main_table(df):
    """Generate Table 3: Strategy-level trade-off."""
    print("% Table 3: Strategy-level trade-off on Qwen3.5-9B")
    # Filter for 9B model, TF strategies
    subset = df[(df["model"].str.contains("9B")) & (df["strategy"].str.contains("_tf|full_think|no_think"))]
    # Group by strategy and compute averages
    for strategy in ["full_think", "no_think", "prompt_tuning", "routing", "speculative"]:
        rows = subset[subset["strategy"].str.contains(strategy)]
        if not rows.empty:
            avg_acc = rows["accuracy"].mean()
            avg_tok = rows["avg_tokens"].mean()
            print(f"  {strategy}: Acc={avg_acc:.1f}%, Tok={avg_tok/1000:.1f}k")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="all_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    generate_main_table(df)


if __name__ == "__main__":
    main()
