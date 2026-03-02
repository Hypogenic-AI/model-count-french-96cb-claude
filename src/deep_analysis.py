"""
Deep analysis of probing results: detailed error patterns, tokenization analysis,
and sensitivity to number structure.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

WORKSPACE = "/workspaces/model-count-french-96cb-claude"
RESULTS_DIR = os.path.join(WORKSPACE, "results")
PLOT_DIR = os.path.join(WORKSPACE, "results/plots")

# Load all data
with open(os.path.join(RESULTS_DIR, "probe_results.json")) as f:
    probe_results = json.load(f)

with open(os.path.join(WORKSPACE, "datasets/french_numbers/french_numbers.json")) as f:
    french_data = json.load(f)

results = probe_results["results"]
test_numbers = probe_results["test_numbers"]
test_vigesimal = probe_results["test_vigesimal"]
formats = probe_results["formats"]
num_layers = probe_results["num_layers"]
best_results = probe_results["best_results"]

colors = {"digits": "#2196F3", "english": "#4CAF50", "french": "#FF5722", "belgian": "#9C27B0"}
labels = {"digits": "Digit strings", "english": "English words",
          "french": "France French", "belgian": "Belgian French"}


def analyze_errors_detailed():
    """Detailed error analysis for all formats."""
    print("="*60)
    print("DETAILED ERROR ANALYSIS")
    print("="*60)

    for fmt in formats:
        preds = np.array(results[fmt]["predictions"])
        trues = np.array(results[fmt]["true"])
        if preds.ndim < 2:
            print(f"\n{fmt}: No predictions available")
            continue

        correct = np.all(preds == trues, axis=1)
        errors = []
        for i in range(len(test_numbers)):
            if not correct[i]:
                n = test_numbers[i]
                entry = french_data[n]
                pred_num = int(preds[i, 0]) * 100 + int(preds[i, 1]) * 10 + int(preds[i, 2])
                errors.append({
                    "true_number": n,
                    "pred_number": pred_num,
                    "true_digits": trues[i].tolist(),
                    "pred_digits": preds[i].tolist(),
                    "french_word": entry["french"],
                    "belgian_word": entry["french_belgian"] or entry["french"],
                    "vigesimal": entry["vigesimal"],
                    "structure": entry["structure"],
                    "error_positions": [j for j in range(3) if preds[i, j] != trues[i, j]],
                })

        print(f"\n{'='*60}")
        print(f"Format: {labels[fmt]} — {len(errors)} errors out of {len(test_numbers)} ({(1-correct.mean())*100:.1f}%)")
        print(f"Best layer: {best_results[fmt]['best_layer']}")
        print(f"{'='*60}")

        for e in errors:
            print(f"  {e['true_number']:3d} ({e['french_word']}) → predicted {e['pred_number']} "
                  f"[{e['true_digits']} → {e['pred_digits']}] "
                  f"{'VIG' if e['vigesimal'] else 'DEC'} "
                  f"structure: {e['structure']}")

    return


def analyze_layer_peak_comparison():
    """Compare where numeric representations peak across formats."""
    print("\n" + "="*60)
    print("LAYER PEAK ANALYSIS")
    print("="*60)

    fig, ax = plt.subplots(figsize=(14, 7))

    for fmt in formats:
        accs = results[fmt]["layer_acc"]
        ax.plot(range(num_layers), accs, color=colors[fmt], label=labels[fmt],
                linewidth=2.5, marker='o', markersize=4)
        peak = np.argmax(accs)
        ax.annotate(f"L{peak}: {accs[peak]:.3f}",
                    xy=(peak, accs[peak]),
                    xytext=(peak + 1.5, accs[peak] + 0.03),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=colors[fmt], lw=1.5),
                    color=colors[fmt])

    ax.set_xlabel("Transformer Layer", fontsize=14)
    ax.set_ylabel("Probe Accuracy (all 3 digits correct)", fontsize=14)
    ax.set_title("When Does Numeric Meaning Emerge?\nLayer-wise Probe Accuracy in Mistral 7B", fontsize=15)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(-0.5, num_layers - 0.5)

    # Shade regions
    ax.axvspan(-0.5, 5.5, alpha=0.08, color='blue', label='_Early layers')
    ax.axvspan(5.5, 15.5, alpha=0.08, color='green', label='_Middle layers')
    ax.axvspan(15.5, num_layers - 0.5, alpha=0.08, color='orange', label='_Late layers')

    ax.text(2.5, -0.04, "Early", ha="center", fontsize=10, color="gray")
    ax.text(10.5, -0.04, "Middle", ha="center", fontsize=10, color="gray")
    ax.text(24, -0.04, "Late", ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "layer_peak_analysis.png"), dpi=150)
    plt.close()
    print("Saved: layer_peak_analysis.png")

    # Print peak layers
    for fmt in formats:
        accs = results[fmt]["layer_acc"]
        peak = np.argmax(accs)
        # Find first layer above 90% accuracy
        above_90 = [i for i, a in enumerate(accs) if a >= 0.9]
        first_90 = above_90[0] if above_90 else "never"
        print(f"  {labels[fmt]:20s}: peak at layer {peak} ({accs[peak]:.3f}), "
              f"first >90% at layer {first_90}")


def analyze_per_digit_layer_combined():
    """Combined per-digit layer-wise analysis."""
    digit_names = ["Hundreds", "Tens", "Units"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, fmt in enumerate(formats):
        per_digit_all = np.array(results[fmt]["per_digit_acc"])
        ax = axes[idx]
        for d_idx, d_name in enumerate(digit_names):
            ax.plot(range(num_layers), per_digit_all[:, d_idx],
                    label=d_name, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Per-Digit Accuracy", fontsize=12)
        ax.set_title(f"{labels[fmt]}", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Per-Digit Accuracy Across Layers\n(Hundreds, Tens, Units positions)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "per_digit_layer_combined.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: per_digit_layer_combined.png")


def analyze_vigesimal_detail():
    """Detailed analysis of vigesimal vs decimal performance."""
    print("\n" + "="*60)
    print("VIGESIMAL VS DECIMAL DETAILED ANALYSIS")
    print("="*60)

    vig_mask = np.array(test_vigesimal)
    dec_mask = ~vig_mask

    # Sub-categorize vigesimal numbers
    vig_70s = np.array([70 <= (n % 100) <= 79 for n in test_numbers])
    vig_80s = np.array([80 <= (n % 100) <= 89 for n in test_numbers])
    vig_90s = np.array([90 <= (n % 100) <= 99 for n in test_numbers])

    for fmt in formats:
        preds = np.array(results[fmt]["predictions"])
        trues = np.array(results[fmt]["true"])
        if preds.ndim < 2:
            continue
        correct = np.all(preds == trues, axis=1)

        total_vig = vig_mask.sum()
        total_dec = dec_mask.sum()
        n_70s = vig_70s.sum()
        n_80s = vig_80s.sum()
        n_90s = vig_90s.sum()

        print(f"\n  {labels[fmt]}:")
        print(f"    Overall: {correct.mean():.3f}")
        print(f"    Decimal (n={total_dec}): {correct[dec_mask].mean():.3f}")
        print(f"    Vigesimal (n={total_vig}): {correct[vig_mask].mean():.3f}")
        if n_70s > 0:
            print(f"      70-79 range (n={n_70s}): {correct[vig_70s].mean():.3f}")
        if n_80s > 0:
            print(f"      80-89 range (n={n_80s}): {correct[vig_80s].mean():.3f}")
        if n_90s > 0:
            print(f"      90-99 range (n={n_90s}): {correct[vig_90s].mean():.3f}")

    # Create detailed vigesimal breakdown bar chart
    categories = ["Decimal\n(0-69)", "70-79\n(soixante-dix)", "80-89\n(quatre-vingt)", "90-99\n(quatre-vingt-dix)"]
    masks = [dec_mask, vig_70s, vig_80s, vig_90s]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.2
    offsets = np.arange(len(formats)) * width - width * (len(formats) - 1) / 2

    for idx, fmt in enumerate(formats):
        preds = np.array(results[fmt]["predictions"])
        trues = np.array(results[fmt]["true"])
        if preds.ndim < 2:
            continue
        correct = np.all(preds == trues, axis=1)

        accs = []
        for mask in masks:
            if mask.sum() > 0:
                accs.append(correct[mask].mean())
            else:
                accs.append(0)

        bars = ax.bar(x + offsets[idx], accs, width, color=colors[fmt],
                      label=labels[fmt], alpha=0.85)
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Number Category", fontsize=13)
    ax.set_ylabel("Probe Accuracy", fontsize=13)
    ax.set_title("Probe Accuracy by Vigesimal Sub-Category\n(France French counting system structure)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "vigesimal_subcategory.png"), dpi=150)
    plt.close()
    print("Saved: vigesimal_subcategory.png")


def analyze_tokenization():
    """Analyze how tokenization correlates with probe accuracy."""
    print("\n" + "="*60)
    print("TOKENIZATION ANALYSIS")
    print("="*60)

    # Token count approximation from the dataset
    test_entries = [french_data[n] for n in test_numbers]
    token_counts = [e["num_tokens_approx"] for e in test_entries]

    for fmt in ["french"]:
        preds = np.array(results[fmt]["predictions"])
        trues = np.array(results[fmt]["true"])
        if preds.ndim < 2:
            continue
        correct = np.all(preds == trues, axis=1)

        # Group by approximate token count
        unique_tokens = sorted(set(token_counts))
        for tc in unique_tokens:
            mask = np.array([t == tc for t in token_counts])
            if mask.sum() > 0:
                acc = correct[mask].mean()
                print(f"  ~{tc} tokens: {mask.sum()} numbers, accuracy={acc:.3f}")


def create_summary_figure():
    """Create a publication-quality summary figure."""
    fig = plt.figure(figsize=(18, 12))

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # Panel A: Layer-wise accuracy
    ax1 = fig.add_subplot(gs[0, :2])
    for fmt in formats:
        accs = results[fmt]["layer_acc"]
        ax1.plot(range(num_layers), accs, color=colors[fmt], label=labels[fmt],
                linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Probe Accuracy", fontsize=12)
    ax1.set_title("A. Layer-wise Probe Accuracy", fontsize=13, fontweight='bold', loc='left')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Panel B: Overall comparison
    ax2 = fig.add_subplot(gs[0, 2])
    overall_accs = [best_results[f]["best_acc"] for f in formats]
    bars = ax2.bar([labels[f] for f in formats], overall_accs,
                  color=[colors[f] for f in formats], alpha=0.85, edgecolor="black")
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel("Best Accuracy", fontsize=12)
    ax2.set_title("B. Best Accuracy per Format", fontsize=13, fontweight='bold', loc='left')
    ax2.set_ylim(0, 1.12)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)

    # Panel C: Vigesimal vs Decimal
    ax3 = fig.add_subplot(gs[1, 0])
    vig_mask = np.array(test_vigesimal)
    dec_mask = ~vig_mask
    x = np.arange(len(formats))
    width = 0.35
    vig_accs = []
    dec_accs = []
    for fmt in formats:
        preds = np.array(results[fmt]["predictions"])
        trues = np.array(results[fmt]["true"])
        correct = np.all(preds == trues, axis=1)
        vig_accs.append(correct[vig_mask].mean())
        dec_accs.append(correct[dec_mask].mean())

    ax3.bar(x - width/2, dec_accs, width, label="Decimal", color="#4CAF50", alpha=0.85)
    ax3.bar(x + width/2, vig_accs, width, label="Vigesimal", color="#FF5722", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([labels[f].split()[-1] for f in formats], fontsize=10)
    ax3.set_ylabel("Accuracy", fontsize=12)
    ax3.set_title("C. Decimal vs Vigesimal", fontsize=13, fontweight='bold', loc='left')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_ylim(0, 1.12)

    # Panel D: Per-digit accuracy for French
    ax4 = fig.add_subplot(gs[1, 1])
    digit_names = ["Hundreds", "Tens", "Units"]
    fr_per_digit = np.array(results["french"]["per_digit_acc"])
    best_l = best_results["french"]["best_layer"]
    per_digit_best = fr_per_digit[best_l]
    ax4.bar(digit_names, per_digit_best, color="#FF5722", alpha=0.85)
    for i, v in enumerate(per_digit_best):
        ax4.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax4.set_ylabel("Accuracy", fontsize=12)
    ax4.set_title(f"D. Per-Digit (French, L{best_l})", fontsize=13, fontweight='bold', loc='left')
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_ylim(0, 1.12)

    # Panel E: French per-digit by layer
    ax5 = fig.add_subplot(gs[1, 2])
    for d_idx, d_name in enumerate(digit_names):
        ax5.plot(range(num_layers), fr_per_digit[:, d_idx], label=d_name, linewidth=2)
    ax5.set_xlabel("Layer", fontsize=12)
    ax5.set_ylabel("Per-Digit Accuracy", fontsize=12)
    ax5.set_title("E. French Per-Digit by Layer", fontsize=13, fontweight='bold', loc='left')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.05, 1.05)

    plt.savefig(os.path.join(PLOT_DIR, "summary_figure.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: summary_figure.png")


def run_sensitivity_check():
    """Run probes at best layers with different random seeds to check stability."""
    import torch

    # Only check the best layer for each format
    from probe_and_analyze import (CircularDigitProbe, load_representations,
                                    number_to_digits, digits_to_circular,
                                    make_stratified_split, NUM_DIGITS, DEVICE)

    with open(os.path.join(RESULTS_DIR, "representations/metadata.json")) as f:
        meta = json.load(f)

    hidden_dim = meta["hidden_dim"]
    numbers = list(range(1000))
    all_digits = np.array([number_to_digits(n) for n in numbers])
    all_circular = np.array([digits_to_circular(number_to_digits(n)) for n in numbers])

    print("\n" + "="*60)
    print("SENSITIVITY CHECK (5 random seeds)")
    print("="*60)

    seed_results = {}
    for fmt in formats:
        best_layer = best_results[fmt]["best_layer"]
        hidden = load_representations(fmt, best_layer)

        accs_per_seed = []
        for seed in [42, 123, 456, 789, 1024]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_idx, test_idx = make_stratified_split(1000, 0.2, seed=seed)
            X_train = hidden[train_idx]
            X_test = hidden[test_idx]
            Y_train = all_circular[train_idx]
            Y_test = all_digits[test_idx]

            probe = CircularDigitProbe(hidden_dim, NUM_DIGITS).to(DEVICE)
            optimizer = torch.optim.Adam(probe.parameters(), lr=5e-4, weight_decay=1e-5)
            loss_fn = torch.nn.MSELoss()

            X_tr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
            Y_tr = torch.tensor(Y_train, dtype=torch.float32).to(DEVICE)
            X_te = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            Y_te = torch.tensor(Y_test, dtype=torch.float32).to(DEVICE)

            probe.train()
            for _ in range(500):
                perm = torch.randperm(len(X_tr))
                for i in range(0, len(X_tr), 128):
                    xb = X_tr[perm[i:i+128]]
                    yb = Y_tr[perm[i:i+128]]
                    pred = probe(xb)
                    loss = loss_fn(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            probe.eval()
            with torch.no_grad():
                pred_digits = probe.predict_digits(X_te)
                pred_rounded = torch.round(pred_digits) % 10
                acc = (pred_rounded == Y_te).all(dim=1).float().mean().item()
            accs_per_seed.append(acc)

        mean_acc = np.mean(accs_per_seed)
        std_acc = np.std(accs_per_seed)
        seed_results[fmt] = {"mean": mean_acc, "std": std_acc, "accs": accs_per_seed}
        print(f"  {labels[fmt]:20s}: {mean_acc:.3f} +/- {std_acc:.3f} "
              f"(seeds: {[f'{a:.3f}' for a in accs_per_seed]})")

    with open(os.path.join(RESULTS_DIR, "sensitivity_results.json"), "w") as f:
        json.dump(seed_results, f, indent=2)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(WORKSPACE, "src"))

    analyze_errors_detailed()
    analyze_layer_peak_comparison()
    analyze_per_digit_layer_combined()
    analyze_vigesimal_detail()
    analyze_tokenization()
    create_summary_figure()
    run_sensitivity_check()

    print("\n" + "="*60)
    print("ALL DEEP ANALYSIS COMPLETE")
    print("="*60)
