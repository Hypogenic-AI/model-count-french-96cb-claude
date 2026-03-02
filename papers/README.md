# Downloaded Papers

Papers relevant to the research question: "How does the model count in French?"

## Core Papers (Number Representations in LLMs)

1. **Language Models Encode Numbers Using Digit Representations in Base 10** (levy_geva_2024_digit_representations_base10.pdf)
   - Authors: Amit Arnold Levy, Mor Geva
   - Year: 2024, arXiv: 2410.11781
   - Why relevant: **FOUNDATIONAL** — shows LLMs use per-digit circular base-10 representations. Primary methodology for our experiments.

2. **Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers** (kadlcik_2025_accurate_number_representations.pdf)
   - Authors: Marek Kadlčík, Michal Štefánik, Timothee Mickus, et al.
   - Year: 2025, arXiv: 2506.08966
   - Why relevant: Extends digit probing to more models with near-perfect accuracy using sinusoidal probes.

3. **What is a Number, That a Large Language Model May Know It?** (marjieh_2025_what_is_a_number.pdf)
   - Authors: Raja Marjieh, Veniamin Veselovsky, Thomas L. Griffiths, Ilia Sucholutsky
   - Year: 2025, arXiv: 2502.01540
   - Why relevant: Shows LLMs blend string-like and numerical representations — relevant for understanding French word-form impacts.

4. **Unravelling the Mechanisms of Manipulating Numbers in Language Models** (stefanik_2025_unravelling_mechanisms_numbers.pdf)
   - Authors: Michal Štefánik, Timothee Mickus, Marek Kadlčík, et al.
   - Year: 2025, arXiv: 2501.03950
   - Why relevant: Universal probes for tracing numerical information through LLM layers.

5. **On Representational Dissociation of Language and Arithmetic in LLMs** (kisako_2025_representational_dissociation.pdf)
   - Authors: Riku Kisako, Tatsuki Kuribayashi, Ryohei Sasano
   - Year: 2025, arXiv: 2411.11627
   - Why relevant: Language and arithmetic occupy separate representational regions — key question for French number words.

## Tokenization and Numeracy

6. **Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs** (singh_strouse_2024_tokenization_counts.pdf)
   - Authors: Aaditya K. Singh, DJ Strouse
   - Year: 2024, arXiv: 2402.14903
   - Why relevant: How tokenization affects numeric reasoning — critical for multi-token French number words.

7. **Scaling Behavior for LLMs Regarding Numeral Systems** (zhou_2024_scaling_numeral_systems.pdf)
   - Authors: Zhejian Zhou, Jiayu Wang, Dahua Lin, Kai Chen
   - Year: 2024, arXiv: 2410.05948
   - Why relevant: Studies base-10 vs base-100/1000 systems — relevant to French vigesimal (partial base-20).

8. **Efficient Numeracy in Language Models Through Single-Token Number Embeddings** (efficient_numeracy_single_token_2025.pdf)
   - Year: 2025, arXiv: 2510.06824
   - Why relevant: Proposes single-token number encodings for better numeracy.

## Arithmetic and Reasoning

9. **NUMCoT: Numerals and Units of Measurement in Chain-of-Thought** (numcot_2024_numerals_units_cot.pdf)
   - Authors: Ancheng Xu et al.
   - Year: 2024, arXiv: 2406.02864, ACL 2024 Findings
   - Why relevant: Studies LLM performance on numeral conversions across systems — directly relevant methodology.

10. **NumeroLogic: Number Encoding for Enhanced LLMs' Numerical Reasoning** (numerologic_2024_number_encoding.pdf)
    - Year: 2024, arXiv: 2404.00459
    - Why relevant: Proposes prefixed notation to aid number understanding.

11. **Investigating the Limitations of Transformers with Simple Arithmetic Tasks** (nogueira_2021_limitations_transformers_arithmetic.pdf)
    - Authors: Rodrigo Nogueira, Zhiying Jiang, Jimmy Lin
    - Year: 2021, arXiv: 2102.13019
    - Why relevant: Foundational work on number surface form effects on arithmetic.

12. **Reverse That Number! Decoding Order Matters in Arithmetic Learning** (zhang_2024_reverse_number_arithmetic.pdf)
    - Authors: Daniel Zhang-Li et al.
    - Year: 2024, arXiv: 2403.05845
    - Why relevant: Digit order matters for arithmetic learning — relevant to French reversed structure.

13. **Semantic Deception: When Reasoning Models Can't Compute an Addition** (nahon_2025_semantic_deception.pdf)
    - Year: 2025, arXiv: 2502.15512
    - Why relevant: Semantic cues disrupt arithmetic — relevant to semantically rich French number words.

## Mechanistic Interpretability

14. **How does GPT-2 Compute Greater-Than?** (hanna_2023_gpt2_greater_than.pdf)
    - Authors: Michael Hanna, Ollie Liu, Alexandre Variengien
    - Year: 2023, arXiv: 2305.00586
    - Why relevant: Template for mechanistic analysis of numerical circuits.

15. **Towards Automated Circuit Discovery for Mechanistic Interpretability** (conmy_2023_automated_circuit_discovery.pdf)
    - Authors: Arthur Conmy et al.
    - Year: 2023, arXiv: 2310.16789
    - Why relevant: ACDC algorithm for automated circuit discovery.

## Benchmarks and Surveys

16. **NumericBench: Exposing Numeracy Gaps** (numericbench_2025_numeracy_gaps.pdf)
    - Year: 2025, arXiv: 2502.11075
    - Why relevant: Comprehensive numeracy benchmark for LLMs.

17. **Can Neural Networks Do Arithmetic? A Survey** (testolin_2023_neural_networks_arithmetic_survey.pdf)
    - Authors: Alberto Testolin
    - Year: 2023, arXiv: 2303.07735
    - Why relevant: Background survey on neural network arithmetic capabilities.

18. **Why Do Large Language Models Struggle to Count Letters?** (llm_struggle_count_letters_2024.pdf)
    - Year: 2024, arXiv: 2412.18626
    - Why relevant: Character-level counting challenges link to tokenization issues.

## Multilingual and Cross-Lingual

19. **Brains and Language Models Converge on a Shared Conceptual Space Across Languages** (zada_2025_brains_lm_shared_conceptual_space.pdf)
    - Authors: Zaid Zada et al.
    - Year: 2025, arXiv: 2407.10223
    - Why relevant: Shows LMs across languages converge on shared meaning — relevant to cross-lingual number representations.

## Numeral Embeddings

20. **Laying Anchors: Semantically Priming Numerals in Language Modeling** (singh_2024_laying_anchors_numerals.pdf)
    - Year: 2024, arXiv: 2404.16130
    - Why relevant: Techniques for improving numeral representations in LMs.

21. **Int2Int: A Framework for Mathematics with Transformers** (charton_2025_int2int_framework.pdf)
    - Authors: Charton
    - Year: 2025, arXiv: 2502.17513
    - Why relevant: Framework for mathematical operations with transformers.
