# Research Plan: How Does the Model Count in French?

## Motivation & Novelty Assessment

### Why This Research Matters
French has a famously counterintuitive counting system that uses vigesimal (base-20) structures for numbers 70-99. For example, 98 = "quatre-vingt-dix-huit" (literally "four-twenty-ten-eight", i.e., 4×20+10+8). Understanding how LLMs internally represent these numbers reveals whether models process number words through their arithmetic structure or treat them as opaque lexical items. This has implications for multilingual NLP, numerical reasoning, and our understanding of how transformers learn mathematical structure from language.

### Gap in Existing Work
Levy & Geva (2024) demonstrated that LLMs encode numbers using base-10 circular digit representations when numbers are presented as digit strings, achieving 91-92% accuracy. They showed partial generalization to English word-form numbers (68.6% for 0-50). However:
- **No study has examined French number word representations**, particularly the vigesimal forms
- **Word-form probing beyond English is unstudied**
- **The natural experiment of vigesimal vs. decimal French variants** (France French vs. Belgian French) has not been exploited
- **Layer-wise emergence of numeric representations from complex word forms** is unexplored

### Our Novel Contribution
We systematically map how an LLM represents French numbers internally by:
1. Extracting hidden-state representations for numbers 0-999 in four formats: digit strings, English words, France French words, Belgian French words
2. Training circular base-10 probes (Levy & Geva, 2024) on each format
3. Comparing probe accuracy across formats, with special attention to vigesimal numbers (70-99)
4. Analyzing layer-wise emergence of numeric representations
5. Performing error analysis to understand if vigesimal structure causes systematic errors

### Experiment Justification
- **Experiment 1 (Digit string baseline)**: Replicates Levy & Geva's findings as a validation baseline, confirming our probing pipeline works correctly.
- **Experiment 2 (English word-form)**: Extends their preliminary word-form results to 0-999, establishing a word-form baseline.
- **Experiment 3 (France French)**: Our core novel experiment — tests whether vigesimal number words can be decoded into base-10 digit representations.
- **Experiment 4 (Belgian French)**: Controlled comparison — same numbers, same language family, but decimal structure instead of vigesimal. Isolates the effect of vigesimal vs. decimal structure.
- **Experiment 5 (Layer-wise analysis)**: Reveals at which transformer layer numeric meaning emerges from French word representations.

## Research Question
Can we map how an LLM internally represents French numbers, and does the vigesimal structure (70-99) affect the quality of these representations compared to decimal equivalents?

## Hypothesis Decomposition

**H1**: LLM hidden representations of French number words encode base-10 digit information that can be recovered by circular probes.

**H2**: Probe accuracy for vigesimal French numbers (70-99, e.g., "quatre-vingt-dix-huit" = 98) is lower than for decimal French numbers (0-69, e.g., "quarante-deux" = 42).

**H3**: Belgian French numbers (decimal: "nonante-huit" = 98) yield higher probe accuracy for 70-99 than France French (vigesimal: "quatre-vingt-dix-huit").

**H4**: Numeric representations emerge at later transformer layers for French word-form numbers than for digit strings.

**H5**: Errors on vigesimal numbers show systematic patterns related to the arithmetic structure (e.g., confusing the base component 4×20=80 with the remainder).

## Proposed Methodology

### Approach
We use the circular probing methodology from Levy & Geva (2024), adapted to work with French number words. For each number 0-999, we:
1. Present the number in a template sentence to the model
2. Extract hidden representations at the last token position across all layers
3. Train circular probes to predict base-10 digits from these representations
4. Compare accuracy across input formats and number ranges

### Model Choice
**Llama 3 8B** — used by Levy & Geva (2024), has individual tokens for 0-999, well-studied representations. We use a single model to keep the comparison clean (same model, different input formats).

### Input Formats
For each number n in [0, 999]:
1. **Digits**: "The number is {n}." → e.g., "The number is 42."
2. **English**: "The number is {english(n)}." → e.g., "The number is forty-two."
3. **France French**: "Le nombre est {french(n)}." → e.g., "Le nombre est quarante-deux."
4. **Belgian French**: "Le nombre est {belgian(n)}." → e.g., "Le nombre est quarante-deux."

### Experimental Steps
1. Load Llama 3 8B and extract hidden representations for all 1000 numbers × 4 formats × all layers
2. Split: 80% train (numbers 0-799), 20% test (800-999) — ensures test set includes vigesimal numbers
3. Train circular probes per layer for each format
4. Evaluate: overall accuracy, per-digit accuracy, vigesimal vs. decimal accuracy
5. Layer-wise analysis: plot accuracy by layer for each format
6. Error analysis: categorize errors by number range and structure

### Baselines
- **Digit string probing**: Expected ~90%+ accuracy (Levy & Geva baseline)
- **Random baseline**: ~10% per digit, ~0.1% for 3-digit exact match
- **English word-form**: Expected ~60-70% (extrapolating from Levy & Geva's 68.6% for 0-50)

### Evaluation Metrics
- **Overall accuracy**: fraction of numbers where all digits are correctly predicted
- **Per-digit accuracy**: accuracy for units, tens, hundreds separately
- **Vigesimal accuracy**: accuracy restricted to numbers with vigesimal components (70-99 in ones/tens place)
- **Decimal accuracy**: accuracy restricted to numbers with decimal components (0-69 in ones/tens place)
- **Layer-wise accuracy curve**: accuracy as a function of transformer layer

### Statistical Analysis Plan
- McNemar's test for paired comparisons (France French vs. Belgian French on same numbers)
- Chi-squared test for vigesimal vs. decimal accuracy differences
- Bootstrap confidence intervals (1000 resamples) for accuracy estimates
- Cohen's h for effect sizes on proportion differences
- Significance level: α = 0.05, with Bonferroni correction for multiple comparisons

## Expected Outcomes
- **Support H1**: French number words contain recoverable numeric information, but at lower accuracy than digit strings
- **Support H2**: Vigesimal numbers (70-99) show 10-20% lower probe accuracy than decimal numbers (0-69)
- **Support H3**: Belgian French achieves 5-15% higher accuracy than France French for 70-99 range
- **Support H4**: Numeric representations peak 2-5 layers later for French words than digit strings
- **Support H5**: Vigesimal errors cluster around arithmetic components (e.g., 98 confused with 78 or 88)

## Timeline and Milestones
1. Environment setup & data prep: 10 min
2. Hidden representation extraction (4 formats × 1000 numbers): 30-45 min
3. Probe training & evaluation: 20-30 min
4. Analysis & visualization: 20-30 min
5. Report writing: 20-30 min

## Potential Challenges
- **Memory**: 4×1000 numbers × 33 layers × 4096 dims is substantial — process in batches
- **Tokenization**: French number words span multiple tokens — must use last-token position consistently
- **Training data overlap**: Some French numbers are likely in the training data — not a concern since we're probing representations, not testing generalization
- **Small test set for vigesimal**: Only 30 vigesimal numbers in 70-99 range — mitigate by including 100s with vigesimal components (170-199, 270-299, etc.)

## Success Criteria
- Digit string probe accuracy >85% (validates our pipeline)
- French word probe accuracy significantly above random (>30%)
- Clear accuracy difference between vigesimal and decimal numbers
- Interpretable layer-wise patterns
- Comprehensive error analysis with actionable insights
