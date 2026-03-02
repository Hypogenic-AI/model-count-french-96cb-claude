# Literature Review: How Does the Model Count in French?

## Research Area Overview

This review covers the intersection of three research areas relevant to understanding how large language models (LLMs) process the French counting system:

1. **Internal number representations in LLMs** — How transformers encode and manipulate numeric information
2. **Tokenization and numeracy** — How tokenization choices affect numerical reasoning
3. **Multilingual and cross-lingual numeric processing** — How number representations interact with language-specific structures

The French counting system is of particular interest because it employs a **vigesimal (base-20)** structure for numbers 70-99, inherited from Gaulish/Celtic linguistic influence. For example, "quatre-vingt-dix-huit" (98) literally means "four-twenty-ten-eight" (4×20+10+8). This creates a natural testbed for understanding whether LLMs process number words through their arithmetic structure or treat them as opaque lexical items.

---

## Key Papers

### Paper 1: Language Models Encode Numbers Using Digit Representations in Base 10
- **Authors**: Amit Arnold Levy, Mor Geva
- **Year**: 2024 (arXiv: 2410.11781)
- **Venue**: arXiv / EMNLP
- **Key Contribution**: Demonstrates that LLMs internally represent numbers using per-digit circular representations in base 10, not as whole values.
- **Methodology**: Circular probing of hidden representations in Llama 3 8B and Mistral 7B. Probes predict individual digit values using circle(t) = [cos(2πt/10), sin(2πt/10)] mappings. Causal interventions on digit representations modify model outputs.
- **Key Results**:
  - Base-10 circular probes achieve 91-92% accuracy in predicting all digits correctly (vs <20% for other bases)
  - In Mistral 7B, best-layer probes achieve **100% accuracy** on validation set
  - LLM errors on arithmetic are scattered across digits (close in "digit space") rather than normally distributed around the correct value
  - Probes partially generalize to word-form numbers (e.g., "twenty-two") with 68.6% peak accuracy
  - Causal interventions on circular digit representations change model outputs as predicted ~50% of the time
- **Code Available**: Yes — https://github.com/amitlevy/base10
- **Relevance to Our Research**: **CRITICAL**. This is the foundational methodology. The key question is: do these base-10 digit representations emerge when numbers are expressed as French words (especially vigesimal forms)? Can we probe "quatre-vingt-dix-huit" and recover [9, 8] as digit representations?

### Paper 2: Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers
- **Authors**: Marek Kadlčík, Michal Štefánik, Timothee Mickus, Michal Spiegel, Josef Kuchar
- **Year**: 2025 (arXiv: 2506.08966)
- **Key Contribution**: Shows that with the right inductive bias (sinusoidal probes), number embeddings can be decoded with near-perfect accuracy across multiple LLM families (Llama 3, Phi 4, OLMo 2, 1B-72B parameters).
- **Key Results**: Embedding precision explains a large portion of arithmetic errors; aligning embeddings can mitigate errors.
- **Relevance**: Confirms that LLMs encode precise numeric information — extends Levy & Geva's findings to more models and scales.

### Paper 3: Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs
- **Authors**: Aaditya K. Singh, DJ Strouse
- **Year**: 2024 (arXiv: 2402.14903)
- **Venue**: arXiv
- **Key Contribution**: Studies how number tokenization (single-digit vs. multi-digit tokens) affects arithmetic performance.
- **Key Results**:
  - Right-to-left tokenization improves arithmetic performance
  - Model errors follow stereotyped patterns suggesting systematic (not approximate) computation
  - Larger models partially overcome tokenization biases
  - 90 citations — highly influential
- **Relevance**: French number words are tokenized very differently from digit strings. Understanding tokenization effects is crucial for our research.

### Paper 4: What is a Number, That a Large Language Model May Know It?
- **Authors**: Raja Marjieh, Veniamin Veselovsky, Thomas L. Griffiths, Ilia Sucholutsky
- **Year**: 2025 (arXiv: 2502.01540)
- **Key Contribution**: Shows LLMs learn representational spaces that blend string-like and numerical representations.
- **Methodology**: Uses similarity-based prompting from cognitive science to elicit similarity judgments over integer pairs.
- **Key Results**: LLM number representations combine Levenshtein edit distance (string similarity) and Log-Linear distance (numerical similarity) — an "entangled representation" that is reflected in latent embeddings.
- **Relevance**: Relevant for understanding how French number words (which have very different string properties from digit strings) may create different representational entanglements.

### Paper 5: NUMCoT: Numerals and Units of Measurement in Chain-of-Thought Reasoning
- **Authors**: Ancheng Xu, Minghuan Tan, Lei Wang, Min Yang, Ruifeng Xu
- **Year**: 2024 (arXiv: 2406.02864)
- **Venue**: ACL 2024 Findings
- **Key Contribution**: Evaluates LLMs on numeral conversions across different numeral systems and units.
- **Methodology**: Constructs perturbed datasets including ancient Chinese arithmetic problems with non-standard numeral systems.
- **Key Results**: LLMs struggle with numeral and measurement conversions, especially when converting between different numeral systems.
- **Code Available**: Yes — https://github.com/CAS-SIAT-XinHai/NUMCoT
- **Relevance**: Directly relevant — demonstrates that numeral system variations (like French vigesimal) challenge LLMs.

### Paper 6: How does GPT-2 Compute Greater-Than?
- **Authors**: Michael Hanna, Ollie Liu, Alexandre Variengien
- **Year**: 2023 (arXiv: 2305.00586)
- **Key Contribution**: Uses mechanistic interpretability to explain GPT-2 small's ability to compare years (e.g., "The war lasted from 1732 to 17__" predicting years > 32).
- **Key Results**: Identifies a circuit of MLPs that boost probabilities of valid end years. The mechanism is general and activates across diverse contexts.
- **Relevance**: Provides methodology template for mechanistic analysis of numerical circuits, applicable to studying French number processing circuits.

### Paper 7: Scaling Behavior for LLMs Regarding Numeral Systems
- **Authors**: Zhejian Zhou, Jiayu Wang, Dahua Lin, Kai Chen
- **Year**: 2024 (arXiv: 2410.05948)
- **Key Contribution**: Studies how different numeral systems (base 10, 100, 1000) affect LLM arithmetic performance.
- **Key Results**: Base 10 is consistently more data-efficient than base 100 or 1000 for from-scratch training, attributed to higher token frequencies. Base 100/1000 systems struggle with token-level discernment.
- **Relevance**: The French vigesimal system is effectively a partial base-20 system. This paper's findings about base effects on learning are directly relevant.

### Paper 8: On Representational Dissociation of Language and Arithmetic in LLMs
- **Authors**: Riku Kisako, Tatsuki Kuribayashi, Ryohei Sasano
- **Year**: 2025 (arXiv: 2411.11627)
- **Key Contribution**: Shows arithmetic equations and general language inputs are encoded in completely separated regions in LLMs' representation space.
- **Key Results**: Simple arithmetic is mapped into a distinct region from general language, including when using spelled-out equations. Suggests a fundamental language-arithmetic dissociation similar to neuroscience findings.
- **Relevance**: Raises the question of whether French number words activate the "language" region or the "arithmetic" region — or both, given their compound arithmetic structure.

### Paper 9: Semantic Deception: When Reasoning Models Can't Compute an Addition
- **Authors**: Nathaniel de Leeuw, M. Nahon, Mathis Reymond, R. Chatila, M. Khamassi
- **Year**: 2025 (arXiv: 2502.15512)
- **Key Contribution**: Shows that semantic cues (misleading symbol associations) significantly degrade LLM performance on simple arithmetic.
- **Key Results**: Even when LLMs correctly follow instructions, semantic cues impact basic capabilities. Chain-of-thought may amplify reliance on statistical correlations.
- **Relevance**: French number words carry strong semantic/linguistic associations that may interfere with numerical processing.

### Paper 10: Can Neural Networks Do Arithmetic? A Survey
- **Authors**: Alberto Testolin
- **Year**: 2023 (arXiv: 2303.07735)
- **Key Contribution**: Comprehensive survey of neural network arithmetic capabilities.
- **Key Results**: Even state-of-the-art architectures often fall short on basic numerical/arithmetic tasks. Identifies key challenges including tokenization, representation, and reasoning.
- **Relevance**: Background survey providing context for the overall research area.

### Paper 11: Investigating the Limitations of Transformers with Simple Arithmetic Tasks
- **Authors**: Rodrigo Nogueira, Zhiying Jiang, Jimmy Lin
- **Year**: 2021 (arXiv: 2102.13019)
- **Key Contribution**: Shows surface form of numbers strongly influences model accuracy on arithmetic.
- **Key Results**: Subword tokenization fails for 5-digit addition. Position tokens (e.g., "3 10e1 2") enable accurate addition up to 60 digits.
- **Code Available**: Yes — https://github.com/castorini/transformers-arithmetic
- **Relevance**: Foundational work on how number surface form affects arithmetic. French number words are an extreme case of "unusual surface form."

### Paper 12: Unravelling the Mechanisms of Manipulating Numbers in Language Models
- **Authors**: Michal Štefánik, Timothee Mickus, Marek Kadlčík, et al.
- **Year**: 2025 (arXiv: 2501.03950)
- **Key Contribution**: Shows different LLMs learn interchangeable, systematic, highly accurate and universal number representations across hidden states and input contexts.
- **Key Results**: Universal probes can be created for each LLM to trace numerical information (including error causes) to specific layers.
- **Relevance**: Universal probing methodology could be applied to French number word contexts.

### Paper 13: Brains and Language Models Converge on a Shared Conceptual Space Across Different Languages
- **Authors**: Zaid Zada, Samuel A. Nastase, Jixing Li, Uri Hasson
- **Year**: 2025 (arXiv: 2407.10223)
- **Key Contribution**: Shows LMs trained on different languages converge onto similar embedding spaces, especially in middle layers. Neural representations of meaning are shared across speakers of English, Chinese, and French.
- **Key Results**: Encoding models trained on one language generalize to predict neural activity in listeners of other languages. Suggests shared meaning emerges despite language diversity.
- **Relevance**: If LLMs have shared meaning representations across languages, French number words should activate similar numeric representations as English number words or digit strings.

---

## Common Methodologies

### Probing Techniques
- **Linear probes**: Predict numeric values from hidden representations (Zhu et al., 2024; Wallace et al., 2019)
- **Circular probes**: Map digits to points on unit circle, achieving near-perfect accuracy (Levy & Geva, 2024; Kadlčík et al., 2025)
- **Sinusoidal probes**: Account for periodic structure of number embeddings (Kadlčík et al., 2025)

### Mechanistic Interpretability
- **Circuit discovery**: Identify minimal circuits responsible for numerical behavior (ACDC, Conmy et al., 2023)
- **Causal interventions**: Modify hidden representations and observe output changes (Levy & Geva, 2024)
- **Activation patching**: Trace information flow through transformer layers

### Evaluation Tasks
- **Multi-operand addition**: Tests arithmetic with controllable complexity
- **Number comparison**: Tests magnitude understanding
- **Numeral conversion**: Tests mapping between different number representations
- **Chain-of-thought probing**: Tests step-by-step numerical reasoning

---

## Standard Baselines

- **Digit-string representation**: Numbers as "123" — the standard baseline for probing
- **English word-form**: Numbers as "one hundred twenty-three" — partially explored by Levy & Geva (2024)
- **Random baseline**: Random probe accuracy (~10% per digit for base 10)
- **Linear probe baseline**: Standard linear regression on hidden states

---

## Evaluation Metrics

- **Probe accuracy**: Fraction of numbers where all digits are correctly predicted
- **Per-digit accuracy**: Accuracy for individual digit positions (units, tens, hundreds)
- **Mean absolute error (MAE)**: For value-level predictions
- **Levenshtein distance**: String-level error metric
- **Causal intervention success rate**: Fraction of interventions producing intended output change

---

## Datasets in the Literature

- **Synthetic arithmetic problems**: Used by most papers (Levy & Geva, Nogueira et al.)
- **NumericBench**: Benchmark for fundamental numerical capabilities (2025)
- **NUMCoT dataset**: Perturbed math word problems with numeral/unit variations
- **BIG-bench**: Contains some numerical reasoning tasks
- **No existing French counting-specific dataset** — this is a gap we address

---

## Gaps and Opportunities

1. **No study of French vigesimal number representations in LLMs**: All existing probing work uses digit strings or simple English word forms. French vigesimal forms (quatre-vingt-dix-huit) have not been studied.

2. **Word-form probing is underdeveloped**: Levy & Geva (2024) show only preliminary results on English word-form numbers (0-50), with partial accuracy. Systematic study of number-word representations is lacking.

3. **Cross-lingual number representation comparison**: While Zada et al. (2025) show shared conceptual spaces across languages, no one has specifically compared numeric representations across languages.

4. **Vigesimal vs. decimal structure analysis**: The French system provides a natural experiment — do numbers 70-99 (vigesimal) show different representational properties than 0-69 (decimal)?

5. **Regional variant comparison**: France French (vigesimal: quatre-vingt-dix) vs. Belgian/Swiss French (decimal: nonante) provides a within-language controlled comparison.

---

## Recommendations for Our Experiment

Based on this literature review:

### Recommended Approach
1. **Use circular digit-wise probes** (Levy & Geva, 2024) as the primary methodology
2. **Compare representations across**: digit strings, English words, France French words, Belgian French words
3. **Focus on numbers 0-999** (matching the range studied in existing work)
4. **Analyze vigesimal vs. decimal numbers** as natural experimental conditions

### Recommended Models
- **Llama 3 8B**: Has individual tokens for 0-999 (used by Levy & Geva)
- **Mistral 7B**: Has single-digit tokenization (different bias, also used by Levy & Geva)
- **A multilingual model** (e.g., BLOOM, mGPT) for comparison

### Recommended Metrics
- Per-digit circular probe accuracy in base 10
- Comparison of probe accuracy: digits vs. English words vs. French words vs. Belgian French
- Error analysis: are errors on vigesimal numbers (70-99) different from decimal numbers?
- Layer-wise analysis: at which layer do French number word representations become "numeric"?

### Methodological Considerations
- French number words will be tokenized into multiple subword tokens — use last-token representation (following Levy & Geva)
- The vigesimal structure creates a natural within-subjects experiment (same model, same numbers, different linguistic form)
- Belgian French provides a crucial control: same numbers, same language, but decimal rather than vigesimal structure
