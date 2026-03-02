# Cloned Repositories

Code repositories relevant to the research on French number representations in LLMs.

## Repo 1: base10
- **URL**: https://github.com/amitlevy/base10
- **Purpose**: Code for the circular digit-wise probing experiment from "Language Models Encode Numbers Using Digit Representations in Base 10" (Levy & Geva, 2024)
- **Location**: code/base10/
- **Key files**:
  - `circular_probe.py` — Core circular probe implementation
  - `train_circ_probes.ipynb` — Notebook for training probes
  - `general_ps_utils/` — General utility functions
- **Notes**: This is the primary codebase to adapt for our experiments. The circular probing methodology will be extended to French number word representations. Currently only includes the main probing experiment code.
- **Dependencies**: PyTorch, transformers (HuggingFace)

## Repo 2: acdc (Automatic Circuit Discovery)
- **URL**: https://github.com/ArthurConmy/Automatic-Circuit-Discovery
- **Purpose**: Automated circuit discovery for mechanistic interpretability (Conmy et al., 2023)
- **Location**: code/acdc/
- **Key files**: See repo README for full structure
- **Notes**: Useful for discovering the circuit responsible for processing French number words vs. digit strings. Can identify which attention heads and MLPs are involved in numerical processing.
- **Dependencies**: TransformerLens, PyTorch

## Repo 3: transformers-arithmetic
- **URL**: https://github.com/castorini/transformers-arithmetic
- **Purpose**: Code for "Investigating the Limitations of Transformers with Simple Arithmetic Tasks" (Nogueira et al., 2021)
- **Location**: code/transformers-arithmetic/
- **Key files**: See repo README
- **Notes**: Baseline code for studying how surface form affects arithmetic. Useful reference for experimental design.
- **Dependencies**: See repo requirements

## Usage Notes for Experiment Runner

The primary code to adapt is `base10`:
1. Extend the probing to accept French number word inputs
2. Compare probe accuracy across: digit strings, English words, France French, Belgian French
3. Analyze per-layer probe accuracy to understand at which processing stage French number words become "numeric"

The `acdc` code is secondary — use it to discover circuits involved in French number processing if time permits.
