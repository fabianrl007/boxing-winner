# Model Card — Boxing Winner v1

**Task:** Binary classification — predict if Opponent 1 wins.  
**Data:** Kaggle fighters.csv + popular_matches.csv (152 fights).  
**Features (Δ = Opp1 − Opp2):** ΔKO%, Δpunch resistance, Δability to take punch, Δavg weight, Δrounds boxed, Δhas-been-KO%, Δpower.  
**Model:** Logistic Regression (scaled, `class_weight='balanced'`).  
**Operating threshold:** 0.55 (tunable 0.50–0.60).

## Performance (held-out test)
- Accuracy: **0.811** (baseline 0.662)  
- ROC-AUC: **0.865**  
- Precision / Recall / F1: **0.714 / 0.769 / 0.741**

## Key Drivers
- ↑ **ΔKO%**, ↑ **ΔResistance**, ↑ **ΔAbility** → higher win odds  
- ↓ **ΔHas-Been-KO%** → lower win odds  
- Weight & rounds: small positive; Power ≈ tied in this dataset

## Limitations
Small dataset (152 fights); power often tied; partial metadata (age/stance/reach); pre-fight aggregates only.

## How to Use
Artifacts in `models/boxing_winner_v1/` (`logreg_delta.pkl`, `config.json`).

```python
from src.predict import predict_winner
opp1 = {"round_ko_percentage": 25.6, "estimated_punch_resistance": 68, "estimated_ability_to_take_punch": 72,
        "avg_weight": 160.0, "estimated_punch_power": 82, "rounds_boxed": 200, "has_been_ko_percentage": 5.0}
opp2 = {"round_ko_percentage": 13.3, "estimated_punch_resistance": 62, "estimated_ability_to_take_punch": 68,
        "avg_weight": 164.3, "estimated_punch_power": 82, "rounds_boxed": 119, "has_been_ko_percentage": 5.0}
predict_winner(opp1, opp2)
