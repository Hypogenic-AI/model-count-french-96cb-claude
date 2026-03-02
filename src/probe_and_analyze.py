"""
Train circular digit probes on extracted representations and analyze results.

For each format (digits, english, french, belgian) and each layer:
- Train a circular probe to predict base-10 digits from hidden states
- Evaluate on held-out test set (random split, stratified by hundreds digit)
- Analyze accuracy by number range (vigesimal vs decimal)
"""

import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

WORKSPACE = "/workspaces/model-count-french-96cb-claude"
REPR_DIR = os.path.join(WORKSPACE, "results/representations")
RESULTS_DIR = os.path.join(WORKSPACE, "results")
PLOT_DIR = os.path.join(WORKSPACE, "results/plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_DIGITS = 3  # hundreds, tens, units for 0-999


class CircularDigitProbe(torch.nn.Module):
    def __init__(self, hidden_dim, num_digits=3, basis=10):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, 2 * num_digits)
        self.num_digits = num_digits
        self.basis = basis

    def forward(self, x):
        proj = self.linear(x)
        return proj.view(-1, self.num_digits, 2)

    def predict_digits(self, x):
        proj = self.forward(x)
        angles = torch.atan2(proj[..., 1], proj[..., 0])
        angles = torch.where(angles < 0, angles + 2 * np.pi, angles)
        digits = angles * self.basis / (2 * np.pi)
        return digits


def number_to_digits(n, num_digits=3):
    digits = []
    for i in range(num_digits - 1, -1, -1):
        digits.append((n // (10 ** i)) % 10)
    return digits


def digits_to_circular(digits, basis=10):
    targets = []
    for d in digits:
        angle = 2 * np.pi * d / basis
        targets.append([np.cos(angle), np.sin(angle)])
    return targets


def load_french_data():
    with open(os.path.join(WORKSPACE, "datasets/french_numbers/french_numbers.json"), "r") as f:
        return json.load(f)


def load_representations(fmt, layer):
    path = os.path.join(REPR_DIR, fmt, f"layer_{layer:02d}.npy")
    return np.load(path)


def make_stratified_split(n_total=1000, test_frac=0.2, seed=42):
    """Create a random train/test split stratified by hundreds digit."""
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    # Stratify by hundreds: 0-99, 100-199, ..., 900-999
    for start in range(0, 1000, 100):
        block = list(range(start, min(start + 100, n_total)))
        rng.shuffle(block)
        n_test = max(1, int(len(block) * test_frac))
        test_idx.extend(block[:n_test])
        train_idx.extend(block[n_test:])
    return sorted(train_idx), sorted(test_idx)


def train_probe(X_train, Y_train_circ, X_test, Y_test_digits, hidden_dim,
                epochs=500, lr=5e-4, batch_size=128):
    """Train a circular probe and return test accuracy and per-digit accuracy."""
    probe = CircularDigitProbe(hidden_dim, NUM_DIGITS).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    Y_tr = torch.tensor(Y_train_circ, dtype=torch.float32).to(DEVICE)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    Y_te_digits = torch.tensor(Y_test_digits, dtype=torch.float32).to(DEVICE)

    probe.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_tr))
        X_tr_s = X_tr[perm]
        Y_tr_s = Y_tr[perm]

        for i in range(0, len(X_tr_s), batch_size):
            xb = X_tr_s[i:i+batch_size]
            yb = Y_tr_s[i:i+batch_size]

            pred = probe(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        pred_digits = probe.predict_digits(X_te)
        pred_rounded = torch.round(pred_digits) % 10

        all_correct = (pred_rounded == Y_te_digits).all(dim=1)
        overall_acc = all_correct.float().mean().item()
        per_digit_acc = (pred_rounded == Y_te_digits).float().mean(dim=0).cpu().numpy()
        pred_np = pred_rounded.cpu().numpy()
        true_np = Y_te_digits.cpu().numpy()

    return overall_acc, per_digit_acc, pred_np, true_np


def run_all_probes():
    """Run probing for all formats and layers."""
    data = load_french_data()

    with open(os.path.join(REPR_DIR, "metadata.json"), "r") as f:
        meta = json.load(f)

    num_layers = meta["num_layers"]
    hidden_dim = meta["hidden_dim"]
    formats = meta["formats"]

    numbers = [entry["number"] for entry in data]
    is_vigesimal = [entry["vigesimal"] for entry in data]

    all_digits = np.array([number_to_digits(n) for n in numbers])
    all_circular = np.array([digits_to_circular(number_to_digits(n)) for n in numbers])

    # Stratified random split
    train_idx, test_idx = make_stratified_split(len(numbers), test_frac=0.2, seed=SEED)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    Y_train_circ = all_circular[train_idx]
    Y_test_digits = all_digits[test_idx]

    test_numbers = [numbers[i] for i in test_idx]
    test_vigesimal = [is_vigesimal[i] for i in test_idx]

    results = {fmt: {"layer_acc": [], "per_digit_acc": [], "predictions": None, "true": None}
               for fmt in formats}
    best_results = {}

    for fmt in formats:
        print(f"\n{'='*60}")
        print(f"Probing format: {fmt}")
        print(f"{'='*60}")

        best_acc = -1
        best_layer = 0

        for layer in tqdm(range(num_layers), desc=f"{fmt} layers"):
            hidden = load_representations(fmt, layer)
            X_train = hidden[train_idx]
            X_test = hidden[test_idx]

            acc, per_digit, pred, true = train_probe(
                X_train, Y_train_circ, X_test, Y_test_digits, hidden_dim,
                epochs=500, lr=5e-4, batch_size=128,
            )

            results[fmt]["layer_acc"].append(acc)
            results[fmt]["per_digit_acc"].append(per_digit.tolist())

            if acc > best_acc:
                best_acc = acc
                best_layer = layer
                results[fmt]["predictions"] = pred.tolist()
                results[fmt]["true"] = true.tolist()

        best_results[fmt] = {"best_layer": best_layer, "best_acc": best_acc}
        print(f"{fmt}: best layer={best_layer}, best acc={best_acc:.4f}")

    output = {
        "formats": formats,
        "num_layers": num_layers,
        "test_numbers": test_numbers,
        "test_vigesimal": test_vigesimal,
        "results": results,
        "best_results": best_results,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }

    with open(os.path.join(RESULTS_DIR, "probe_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    print("\nAll probing complete.")
    return output


def analyze_results():
    """Analyze probe results comprehensively."""
    with open(os.path.join(RESULTS_DIR, "probe_results.json"), "r") as f:
        output = json.load(f)

    results = output["results"]
    test_numbers = output["test_numbers"]
    test_vigesimal = output["test_vigesimal"]
    formats = output["formats"]
    num_layers = output["num_layers"]
    best_results = output["best_results"]

    colors = {"digits": "#2196F3", "english": "#4CAF50", "french": "#FF5722", "belgian": "#9C27B0"}
    labels = {"digits": "Digit strings", "english": "English words",
              "french": "France French", "belgian": "Belgian French"}

    # ── 1. Layer-wise accuracy plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for fmt in formats:
        accs = results[fmt]["layer_acc"]
        ax.plot(range(num_layers), accs, color=colors[fmt], label=labels[fmt], linewidth=2)
        best_l = best_results[fmt]["best_layer"]
        best_a = best_results[fmt]["best_acc"]
        ax.scatter(best_l, best_a, color=colors[fmt], s=80, zorder=5, edgecolors="black")

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Probe Accuracy (all 3 digits correct)", fontsize=13)
    ax.set_title("Layer-wise Circular Probe Accuracy: How Mistral 7B Represents Numbers\nacross Digit Strings, English, and French Formats", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "layer_wise_accuracy.png"), dpi=150)
    plt.close()
    print("Saved: layer_wise_accuracy.png")

    # ── 2. Per-digit accuracy at best layer ──────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    digit_names = ["Hundreds", "Tens", "Units"]

    for idx, fmt in enumerate(formats):
        best_l = best_results[fmt]["best_layer"]
        per_digit = results[fmt]["per_digit_acc"][best_l]
        axes[idx].bar(digit_names, per_digit, color=colors[fmt], alpha=0.85)
        axes[idx].set_title(f"{labels[fmt]}\n(Layer {best_l})", fontsize=12)
        axes[idx].set_ylim(0, 1.15)
        axes[idx].grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(per_digit):
            axes[idx].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    axes[0].set_ylabel("Per-Digit Accuracy", fontsize=12)
    fig.suptitle("Per-Digit Probe Accuracy at Best Layer", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "per_digit_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: per_digit_accuracy.png")

    # ── 3. Vigesimal vs Decimal accuracy at best layer ───────────────────────
    vig_mask = np.array(test_vigesimal)
    dec_mask = ~vig_mask

    vig_dec_results = {}
    for fmt in formats:
        preds = np.array(results[fmt]["predictions"])
        trues = np.array(results[fmt]["true"])
        if preds.ndim < 2:
            vig_dec_results[fmt] = {"vigesimal_acc": 0, "decimal_acc": 0, "overall_acc": 0,
                                    "n_vigesimal": int(vig_mask.sum()), "n_decimal": int(dec_mask.sum())}
            continue

        correct = np.all(preds == trues, axis=1)
        vig_acc = correct[vig_mask].mean() if vig_mask.sum() > 0 else 0
        dec_acc = correct[dec_mask].mean() if dec_mask.sum() > 0 else 0
        overall_acc = correct.mean()

        vig_dec_results[fmt] = {
            "vigesimal_acc": float(vig_acc),
            "decimal_acc": float(dec_acc),
            "overall_acc": float(overall_acc),
            "n_vigesimal": int(vig_mask.sum()),
            "n_decimal": int(dec_mask.sum()),
        }
        print(f"{fmt}: overall={overall_acc:.3f}, vigesimal={vig_acc:.3f} (n={vig_mask.sum()}), "
              f"decimal={dec_acc:.3f} (n={dec_mask.sum()})")

    # Bar chart: vigesimal vs decimal
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(formats))
    width = 0.35

    vig_accs = [vig_dec_results[f]["vigesimal_acc"] for f in formats]
    dec_accs = [vig_dec_results[f]["decimal_acc"] for f in formats]

    bars1 = ax.bar(x - width/2, dec_accs, width, label="Decimal (0-69 remainder)",
                   color="#4CAF50", alpha=0.85)
    bars2 = ax.bar(x + width/2, vig_accs, width, label="Vigesimal (70-99 remainder)",
                   color="#FF5722", alpha=0.85)

    ax.set_xlabel("Input Format", fontsize=13)
    ax.set_ylabel("Probe Accuracy", fontsize=13)
    ax.set_title("Probe Accuracy: Decimal vs. Vigesimal Numbers\n(at best layer per format)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[f] for f in formats], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "vigesimal_vs_decimal.png"), dpi=150)
    plt.close()
    print("Saved: vigesimal_vs_decimal.png")

    # ── 4. Overall comparison bar chart ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    overall_accs = [best_results[f]["best_acc"] for f in formats]
    best_layers = [best_results[f]["best_layer"] for f in formats]

    bars = ax.bar([labels[f] for f in formats], overall_accs,
                  color=[colors[f] for f in formats], alpha=0.85, edgecolor="black")
    for bar, layer in zip(bars, best_layers):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.2f}\n(L{layer})', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel("Best Probe Accuracy (all digits correct)", fontsize=13)
    ax.set_title("Overall Probe Accuracy by Input Format", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "overall_comparison.png"), dpi=150)
    plt.close()
    print("Saved: overall_comparison.png")

    # ── 5. Detailed per-digit layer-wise heatmaps ───────────────────────────
    digit_names = ["Hundreds", "Tens", "Units"]
    for fmt in formats:
        per_digit_all = np.array(results[fmt]["per_digit_acc"])  # (num_layers, 3)
        fig, ax = plt.subplots(figsize=(8, 6))
        for d_idx, d_name in enumerate(digit_names):
            ax.plot(range(num_layers), per_digit_all[:, d_idx], label=d_name, linewidth=2)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Per-Digit Accuracy", fontsize=12)
        ax.set_title(f"Per-Digit Accuracy by Layer: {labels[fmt]}", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"per_digit_layer_{fmt}.png"), dpi=150)
        plt.close()
    print("Saved: per_digit_layer_*.png")

    # ── 6. Error analysis for French format ──────────────────────────────────
    fr_preds = np.array(results["french"]["predictions"])
    fr_trues = np.array(results["french"]["true"])

    if fr_preds.ndim >= 2:
        fr_correct = np.all(fr_preds == fr_trues, axis=1)
        pred_numbers = fr_preds[:, 0] * 100 + fr_preds[:, 1] * 10 + fr_preds[:, 2]
        true_numbers_arr = np.array(test_numbers)
        errors = pred_numbers - true_numbers_arr
        error_mask = ~fr_correct

        if error_mask.sum() > 0:
            error_magnitudes = np.abs(errors[error_mask])
            error_values = errors[error_mask]
            error_true = true_numbers_arr[error_mask]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(error_magnitudes, bins=30, color="#FF5722", alpha=0.75, edgecolor="black")
            axes[0].set_xlabel("Absolute Error", fontsize=12)
            axes[0].set_ylabel("Count", fontsize=12)
            axes[0].set_title("Distribution of Error Magnitudes (France French)", fontsize=13)
            axes[0].grid(True, alpha=0.3)

            vig_err_mask = vig_mask[error_mask]
            dec_err_mask = dec_mask[error_mask]
            axes[1].scatter(error_true[vig_err_mask], error_values[vig_err_mask],
                           color="#FF5722", alpha=0.5, label="Vigesimal", s=20)
            axes[1].scatter(error_true[dec_err_mask], error_values[dec_err_mask],
                           color="#4CAF50", alpha=0.5, label="Decimal", s=20)
            axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
            axes[1].set_xlabel("True Number", fontsize=12)
            axes[1].set_ylabel("Prediction Error", fontsize=12)
            axes[1].set_title("Prediction Errors by True Value (France French)", fontsize=13)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "error_analysis_french.png"), dpi=150)
            plt.close()
            print("Saved: error_analysis_french.png")

            # Vigesimal-specific error patterns
            vig_errors = []
            for i in range(len(test_numbers)):
                if vig_mask[i] and not fr_correct[i]:
                    vig_errors.append({
                        "true_number": int(test_numbers[i]),
                        "pred_digits": fr_preds[i].tolist(),
                        "true_digits": fr_trues[i].tolist(),
                        "error_positions": [j for j in range(3) if fr_preds[i][j] != fr_trues[i][j]],
                    })

            digit_error_counts = [0, 0, 0]
            for e in vig_errors:
                for pos in e["error_positions"]:
                    digit_error_counts[pos] += 1
            print(f"Vigesimal error digit positions (H/T/U): {digit_error_counts}")
            print(f"Total vigesimal errors: {len(vig_errors)}")

    # ── 7. Statistical tests ────────────────────────────────────────────────
    stats_results = {}

    # McNemar's test: French vs Belgian for vigesimal numbers
    for fmt1, fmt2 in [("french", "belgian"), ("french", "english"), ("french", "digits")]:
        p1 = np.array(results[fmt1]["predictions"])
        t1 = np.array(results[fmt1]["true"])
        p2 = np.array(results[fmt2]["predictions"])
        t2 = np.array(results[fmt2]["true"])
        if p1.ndim < 2 or p2.ndim < 2:
            continue

        c1 = np.all(p1 == t1, axis=1)
        c2 = np.all(p2 == t2, axis=1)
        b = np.sum(c1 & ~c2)
        c = np.sum(~c1 & c2)

        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat, mcnemar_p = 0.0, 1.0

        stats_results[f"mcnemar_{fmt1}_vs_{fmt2}"] = {
            "statistic": float(mcnemar_stat),
            "p_value": float(mcnemar_p),
            f"{fmt1}_right_{fmt2}_wrong": int(b),
            f"{fmt1}_wrong_{fmt2}_right": int(c),
        }
        print(f"McNemar ({fmt1} vs {fmt2}): stat={mcnemar_stat:.3f}, p={mcnemar_p:.4f}")

    # Chi-squared: vigesimal vs decimal for French
    if "french" in results and np.array(results["french"]["predictions"]).ndim >= 2:
        fr_all_correct = np.all(np.array(results["french"]["predictions"]) ==
                                np.array(results["french"]["true"]), axis=1)
        vig_correct = fr_all_correct[vig_mask].sum()
        vig_total = vig_mask.sum()
        dec_correct = fr_all_correct[dec_mask].sum()
        dec_total = dec_mask.sum()

        if vig_total > 0 and dec_total > 0:
            contingency = np.array([[vig_correct, vig_total - vig_correct],
                                    [dec_correct, dec_total - dec_correct]])
            # Use Fisher's exact test if any cell is small
            if contingency.min() < 5:
                odds, fisher_p = stats.fisher_exact(contingency)
                stats_results["fisher_vigesimal_vs_decimal_french"] = {
                    "odds_ratio": float(odds),
                    "p_value": float(fisher_p),
                    "vigesimal_acc": float(vig_correct / vig_total),
                    "decimal_acc": float(dec_correct / dec_total),
                }
                print(f"Fisher exact (vig vs dec, French): OR={odds:.3f}, p={fisher_p:.4f}")
            else:
                chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)
                p1 = vig_correct / vig_total
                p2 = dec_correct / dec_total
                cohens_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
                stats_results["chi2_vigesimal_vs_decimal_french"] = {
                    "chi2": float(chi2),
                    "p_value": float(chi2_p),
                    "vigesimal_acc": float(p1),
                    "decimal_acc": float(p2),
                    "cohens_h": float(cohens_h),
                }
                print(f"Chi2 (vig vs dec, French): chi2={chi2:.3f}, p={chi2_p:.4f}")

    # Bootstrap CI for best accuracies
    n_boot = 1000
    bootstrap_cis = {}
    for fmt in formats:
        preds = np.array(results[fmt]["predictions"])
        trues = np.array(results[fmt]["true"])
        if preds.ndim < 2:
            continue
        correct = np.all(preds == trues, axis=1)
        boot_accs = []
        rng = np.random.RandomState(SEED)
        for _ in range(n_boot):
            idx = rng.choice(len(correct), len(correct), replace=True)
            boot_accs.append(correct[idx].mean())
        ci_low, ci_high = np.percentile(boot_accs, [2.5, 97.5])
        bootstrap_cis[fmt] = {"mean": float(correct.mean()), "ci_low": float(ci_low),
                              "ci_high": float(ci_high)}
        print(f"{fmt}: acc={correct.mean():.3f} [{ci_low:.3f}, {ci_high:.3f}]")

    # ── 8. Accuracy heatmap by number range ─────────────────────────────────
    range_size = 50
    range_starts = list(range(0, 1000, range_size))
    heatmap_data = {}

    for fmt in formats:
        preds = np.array(results[fmt]["predictions"])
        trues = np.array(results[fmt]["true"])
        if preds.ndim < 2:
            heatmap_data[fmt] = [0] * len(range_starts)
            continue

        correct = np.all(preds == trues, axis=1)
        range_accs = []
        for start in range_starts:
            end = start + range_size
            mask = np.array([(start <= n < end) for n in test_numbers])
            if mask.sum() > 0:
                range_accs.append(correct[mask].mean())
            else:
                range_accs.append(float('nan'))
        heatmap_data[fmt] = range_accs

    fig, ax = plt.subplots(figsize=(16, 4))
    hm = np.array([heatmap_data[f] for f in formats])
    range_labels = [f"{s}-{s+range_size-1}" for s in range_starts]

    im = ax.imshow(hm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(formats)))
    ax.set_yticklabels([labels[f] for f in formats], fontsize=11)
    ax.set_xticks(range(len(range_labels)))
    ax.set_xticklabels(range_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Number Range", fontsize=12)
    ax.set_title("Probe Accuracy Heatmap by Number Range", fontsize=13)
    plt.colorbar(im, ax=ax, label="Accuracy")

    for i in range(len(formats)):
        for j in range(len(range_labels)):
            if not np.isnan(hm[i, j]):
                ax.text(j, i, f"{hm[i, j]:.2f}", ha="center", va="center",
                        color="black" if hm[i, j] > 0.5 else "white", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "accuracy_heatmap.png"), dpi=150)
    plt.close()
    print("Saved: accuracy_heatmap.png")

    # ── Save all analysis ────────────────────────────────────────────────────
    analysis = {
        "best_results": best_results,
        "vigesimal_decimal": vig_dec_results,
        "statistical_tests": stats_results,
        "bootstrap_ci": bootstrap_cis,
    }
    with open(os.path.join(RESULTS_DIR, "analysis_results.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    print("\nAll analysis complete.")
    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if args.analyze_only:
        analyze_results()
    elif args.probe_only:
        run_all_probes()
    else:
        run_all_probes()
        analyze_results()
