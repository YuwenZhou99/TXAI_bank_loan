# XAI Bank Loan Project

This project studies fairness and explainable AI using the Fathom dataset from https://www.kaggle.com/datasets/udaymalviya/bank-loan-data/data

## Setup
Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the models

Run baseline model with original dataset:
```bash
python src/xai/shap_analysis.py --mode original
```

Run baseline model with downsampled dataset:
```bash
python src/xai/shap_analysis.py --mode downsampled
```

Run fairness model with demographic parity:
```bash
python src/xai/shap_analysis.py --mode fairness --constraint demographic_parity
```

Run fairness model with equalized odds:
```bash
python src/xai/shap_analysis.py --mode fairness --constraint equalized_odds
```

## Authors
- Yuwen Zhou (y.zhou.74@student.rug.nl)
- Eli Baho (e.p.baho@student.rug.nl)
- Marco Capoccia (m.capoccia@student.rug.nl)
