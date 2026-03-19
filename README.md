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

## Individual contributions

We discussed all tasks in advance, divided them as evenly as possible among the group members, and all members agreed on the final distribution of work shown in the table below. Therefore, we believe all group members contributed equally to this assignment.

| Name | Main Tasks | Other |
|---|---|---|
| Eli Baho | Methodology, Findings Subsections, Limitations | Review other tasks |
| Marco Capoccia | Abstract, Introduction, Case Study, Analysis | Review other tasks |
| Yuwen Zhou | Code Work, Plots, Experimental Setup | Review other tasks |
