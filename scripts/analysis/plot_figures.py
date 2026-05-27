"""
Generate paper figures from collected results.
Usage: python scripts/analysis/plot_figures.py --csv all_results.csv --output_dir figures/
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pareto_front(df, output_dir):
    """Plot efficiency-effectiveness Pareto front (Figure 2)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    strategy_colors = {"prompt_tuning": "#FF4444", "routing": "#2196F3", "speculative": "#00BFA5"}
    
    for strategy, color in strategy_colors.items():
        subset = df[df["strategy"].str.contains(strategy)]
        if not subset.empty:
            ax.scatter(subset["avg_tokens"], subset["accuracy"], 
                      c=color, label=strategy.replace("_", " ").title(), s=120)
    
    ax.set_xlabel("Average Tokens", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Efficiency-Effectiveness Trade-off", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pareto_front.pdf", dpi=300)
    plt.savefig(f"{output_dir}/pareto_front.png", dpi=150)
    print(f"Saved pareto_front to {output_dir}")


def plot_training_effect(df, output_dir):
    """Plot training effect comparison (Figure 4)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Accuracy panel and Efficiency panel
    # ... (customize based on data)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_effect.pdf", dpi=300)
    plt.savefig(f"{output_dir}/training_effect.png", dpi=150)
    print(f"Saved training_effect to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="all_results.csv")
    parser.add_argument("--output_dir", default="figures/")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(args.csv)
    plot_pareto_front(df, args.output_dir)
    plot_training_effect(df, args.output_dir)


if __name__ == "__main__":
    main()
