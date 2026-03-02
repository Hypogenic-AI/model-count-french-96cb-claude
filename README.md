# How Does the Model Count in French?

Mapping internal number representations of the French vigesimal counting system in LLMs.

## Key Findings

- **98% accuracy** decoding French number words into base-10 digits from Mistral 7B hidden states — including vigesimal forms like "quatre-vingt-dix-huit" (4×20+10+8 = 98)
- **Word-form numbers peak at early layers (5-6)**, while digit strings peak at the final layer (31) — suggesting fundamentally different processing pathways
- **Vigesimal structure leaves fingerprints**: errors on 73 and 76 reveal the model partially encodes "soixante" (60) rather than the composed value (70+)
- **Belgian French** (decimal: septante, huitante, nonante) achieves **99% accuracy** and **100% on vigesimal range** — the simpler naming system produces cleaner representations
- **French significantly outperforms digit strings** (98% vs 91%, p=0.002), suggesting semantic processing extracts numeric meaning more cleanly than token-level processing

## Quick Start

```bash
# Setup environment
source .venv/bin/activate

# Extract hidden representations (requires GPU)
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache
python src/extract_representations.py

# Train probes and analyze
python src/probe_and_analyze.py

# Deep analysis (error patterns, sensitivity, publication figures)
python src/deep_analysis.py
```

## File Structure

```
├── REPORT.md                  # Full research report with results
├── README.md                  # This file
├── planning.md                # Research plan and motivation
├── src/
│   ├── extract_representations.py  # Extract hidden states from Mistral 7B
│   ├── probe_and_analyze.py        # Train circular probes and analyze
│   └── deep_analysis.py            # Detailed error analysis and figures
├── datasets/
│   └── french_numbers/
│       ├── french_numbers.json     # 0-999 with France + Belgian French
│       └── generate_french_numbers.py
├── results/
│   ├── representations/       # Extracted hidden states (per layer, per format)
│   ├── probe_results.json     # Full probing results
│   ├── analysis_results.json  # Statistical analysis
│   ├── sensitivity_results.json
│   └── plots/                 # All visualizations
│       ├── summary_figure.png
│       ├── layer_peak_analysis.png
│       ├── vigesimal_subcategory.png
│       └── ...
├── papers/                    # Reference papers (PDFs)
├── code/                      # Reference code repositories
├── literature_review.md       # Background literature
└── resources.md               # Resource catalog
```

## Method

We use **circular digit probes** (Levy & Geva, 2024) to decode base-10 digits from transformer hidden states. For each number 0-999 in four formats (digit strings, English words, France French, Belgian French), we extract last-token hidden states across all 33 layers of Mistral 7B, then train linear probes mapping hidden states to circular encodings of hundreds/tens/units digits.

## Requirements

- Python 3.12+
- PyTorch 2.10+ with CUDA
- transformers 5.2+
- NVIDIA GPU with 16GB+ VRAM (tested on RTX A6000)

See full details in [REPORT.md](REPORT.md).
