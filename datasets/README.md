# Datasets

This directory contains datasets for the research project "How does the model count in French?"

Data files are excluded from git (except samples). Use the generation scripts or download instructions below.

## Dataset 1: French Numbers (0-999)

### Overview
- **Source**: Generated locally using `french_numbers/generate_french_numbers.py`
- **Size**: 1,000 entries (numbers 0-999)
- **Format**: JSON
- **Task**: Probing LLM internal representations of French number words
- **License**: Created for this project

### Generation Instructions

```bash
cd datasets/french_numbers
python generate_french_numbers.py
```

This creates `french_numbers.json` with all 1,000 entries.

### Loading the Dataset

```python
import json
with open("datasets/french_numbers/french_numbers.json") as f:
    data = json.load(f)
```

### Schema

Each entry contains:
```json
{
  "number": 98,
  "french": "quatre-vingt-dix-huit",
  "french_belgian": "nonante-huit",
  "digits": "98",
  "structure": "4*20+10+8",
  "vigesimal": true,
  "num_tokens_approx": 4
}
```

### Key Statistics
- 1,000 total entries (0-999)
- 300 numbers use vigesimal (base-20) components (numbers where last two digits are 70-99)
- 300 entries have Belgian/Swiss French variants

### Experimental Conditions

The dataset naturally divides into conditions for analysis:

| Condition | Numbers | Count | Description |
|-----------|---------|-------|-------------|
| Decimal | 0-69, X00-X69 | 700 | Standard base-10 structure |
| Vigesimal 70s | X70-X79 | 100 | soixante-dix (60+10) system |
| Vigesimal 80s | X80-X89 | 100 | quatre-vingts (4×20) system |
| Vigesimal 90s | X90-X99 | 100 | quatre-vingt-dix (4×20+10) system |

### Notes
- Uses 1990 French orthographic reform (hyphens throughout compound numerals)
- France French (standard) uses vigesimal forms for 70-99
- Belgian/Swiss French uses decimal forms (septante, huitante/octante, nonante)
- Both variants included for controlled comparison experiments
