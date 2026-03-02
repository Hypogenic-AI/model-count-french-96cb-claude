# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project: **"How does the model count in French?"**

The research investigates how LLMs internally represent and process the French counting system, which uses a vigesimal (base-20) structure for numbers 70-99.

### Papers
Total papers downloaded: 21

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Language Models Encode Numbers Using Digit Representations in Base 10 | Levy, Geva | 2024 | papers/levy_geva_2024_*.pdf | **Core methodology** — circular digit probes |
| 2 | Pre-trained LMs Learn Remarkably Accurate Representations of Numbers | Kadlčík et al. | 2025 | papers/kadlcik_2025_*.pdf | Near-perfect number probing |
| 3 | What is a Number, That an LLM May Know It? | Marjieh et al. | 2025 | papers/marjieh_2025_*.pdf | String-number entangled representations |
| 4 | Unravelling Mechanisms of Manipulating Numbers in LMs | Štefánik et al. | 2025 | papers/stefanik_2025_*.pdf | Universal number probes |
| 5 | On Representational Dissociation of Language and Arithmetic | Kisako et al. | 2025 | papers/kisako_2025_*.pdf | Language-arithmetic separation |
| 6 | Tokenization Counts | Singh, Strouse | 2024 | papers/singh_strouse_2024_*.pdf | Tokenization effects on arithmetic |
| 7 | Scaling Behavior for LLMs Regarding Numeral Systems | Zhou et al. | 2024 | papers/zhou_2024_*.pdf | Base-10 vs other numeral systems |
| 8 | Efficient Numeracy Through Single-Token Embeddings | — | 2025 | papers/efficient_numeracy_*.pdf | Single-token number encodings |
| 9 | NUMCoT: Numerals and Units in Chain-of-Thought | Xu et al. | 2024 | papers/numcot_2024_*.pdf | Cross-system numeral evaluation |
| 10 | NumeroLogic: Number Encoding for Enhanced Reasoning | — | 2024 | papers/numerologic_2024_*.pdf | Number encoding methods |
| 11 | Investigating Limitations of Transformers with Arithmetic | Nogueira et al. | 2021 | papers/nogueira_2021_*.pdf | Surface form effects |
| 12 | Reverse That Number | Zhang-Li et al. | 2024 | papers/zhang_2024_*.pdf | Digit order in arithmetic |
| 13 | Semantic Deception | de Leeuw et al. | 2025 | papers/nahon_2025_*.pdf | Semantic interference in arithmetic |
| 14 | How does GPT-2 Compute Greater-Than? | Hanna et al. | 2023 | papers/hanna_2023_*.pdf | Mechanistic interpretability of numbers |
| 15 | Towards Automated Circuit Discovery | Conmy et al. | 2023 | papers/conmy_2023_*.pdf | ACDC algorithm |
| 16 | NumericBench: Exposing Numeracy Gaps | — | 2025 | papers/numericbench_2025_*.pdf | Numeracy benchmark |
| 17 | Can Neural Networks Do Arithmetic? Survey | Testolin | 2023 | papers/testolin_2023_*.pdf | Comprehensive survey |
| 18 | Why Do LLMs Struggle to Count Letters? | — | 2024 | papers/llm_struggle_count_letters_*.pdf | Counting challenges |
| 19 | Brains and LMs Converge on Shared Conceptual Space | Zada et al. | 2025 | papers/zada_2025_*.pdf | Cross-lingual representations |
| 20 | Laying Anchors: Semantically Priming Numerals | Singh et al. | 2024 | papers/singh_2024_*.pdf | Numeral embeddings |
| 21 | Int2Int: Framework for Mathematics with Transformers | Charton | 2025 | papers/charton_2025_*.pdf | Math transformer framework |

See papers/README.md for detailed descriptions.

### Datasets
Total datasets created: 1

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| French Numbers 0-999 | Generated | 1,000 entries, ~200KB | Probing | datasets/french_numbers/ | Custom dataset with France & Belgian French |

See datasets/README.md for detailed descriptions.

### Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| base10 | github.com/amitlevy/base10 | Circular digit probes for LLM number representations | code/base10/ | **Primary code to adapt** |
| acdc | github.com/ArthurConmy/Automatic-Circuit-Discovery | Automated circuit discovery for mech. interp. | code/acdc/ | Secondary — for circuit analysis |
| transformers-arithmetic | github.com/castorini/transformers-arithmetic | Surface form effects on transformer arithmetic | code/transformers-arithmetic/ | Reference implementation |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. **Paper-finder service** with diligent mode for three query categories: LLM numeracy, French multilingual processing, mechanistic interpretability of numbers
2. **Web search** targeting recent (2024-2025) work on LLM number representations, French language processing, and cross-lingual numeracy
3. **Semantic Scholar API** results from paper-finder for citation-ranked papers
4. **GitHub search** for code from key papers

### Selection Criteria
- Papers directly studying how LLMs represent numbers internally (probing, interpretability)
- Papers on tokenization effects on numerical reasoning
- Papers on multilingual/cross-lingual number processing
- Papers on numeral system effects on LLM performance
- Comprehensive surveys of the field

### Challenges Encountered
1. **No existing research directly on French counting in LLMs**: This is a genuine research gap. No published work specifically studies how LLMs handle the vigesimal French number system.
2. **Wrong arXiv IDs from search results**: Some Semantic Scholar corpus IDs did not map directly to correct arXiv IDs, requiring web search verification.
3. **No existing French number dataset**: Required creating a custom dataset from scratch.

### Gaps and Workarounds
1. **No French-specific numeracy dataset** → Created comprehensive French numbers dataset (0-999) with both France and Belgian variants
2. **No direct study of vigesimal representations** → Literature on base effects (Zhou et al., 2024) and numeral system conversions (NUMCoT) provides relevant methodology
3. **Limited word-form probing literature** → Levy & Geva (2024) show preliminary word-form results (68.6% accuracy for English 0-50), which our work can build on

---

## Recommendations for Experiment Design

Based on gathered resources, we recommend:

### 1. Primary Dataset
**French Numbers (0-999)** — the custom dataset in `datasets/french_numbers/`
- Covers the full range where Levy & Geva have established base-10 digit representations
- Includes both vigesimal (France French) and decimal (Belgian French) variants
- Natural experimental conditions: decimal (0-69) vs. vigesimal (70-99)

### 2. Baseline Methods
- **Digit string probing**: Replicate Levy & Geva (2024) results as baseline
- **English word-form probing**: Extend their preliminary word-form results
- **French word-form probing**: Our novel contribution
- **Belgian French probing**: Controlled comparison (same numbers, decimal structure)

### 3. Evaluation Metrics
- Per-digit circular probe accuracy (base 10)
- Overall number reconstruction accuracy
- Layer-wise accuracy profiles
- Error analysis: vigesimal vs. decimal numbers
- Comparison across input formats (digits, English, France French, Belgian French)

### 4. Code to Adapt/Reuse
- **`code/base10/`**: Primary codebase — extend `circular_probe.py` and `train_circ_probes.ipynb` to accept French word inputs
- **`code/acdc/`**: Secondary — for circuit discovery if time permits

### 5. Models to Test
- **Llama 3 8B**: Individual tokens for 0-999, used in foundational studies
- **Mistral 7B**: Single-digit tokenization, provides contrast
- Consider a multilingual model (e.g., BLOOM, Croissant) for comparison

### 6. Key Experimental Questions
1. Can circular base-10 digit probes recover the numeric value from French number word representations?
2. Is probe accuracy lower for vigesimal numbers (70-99) than decimal numbers (0-69)?
3. At which transformer layer do French number words become "numeric"?
4. Do Belgian French (decimal) words yield better probe accuracy than France French (vigesimal) words for 70-99?
5. What error patterns emerge? Do vigesimal number errors reflect the arithmetic structure (e.g., errors in the "vingt" component)?
