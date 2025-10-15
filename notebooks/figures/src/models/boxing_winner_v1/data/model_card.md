# Model Card — Boxing Winner v1

**Task:** Binary classification (Opponent 1 wins?).  
**Data:** Kaggle fighters.csv + popular_matches.csv (152 fights).  
**Features:** Δ (Opponent1 − Opponent2) of KO%, punch resistance, ability to take punch, avg weight, rounds boxed, has-been-KO%, power.  
**Model:** Logistic Regression (scaled, class_weight='balanced').  
** Operating threshold:** 0.55 (tunable 0.50–0.60).

## Performance (held-out test)
- Accuracy: 0.811 (baseline 0.662)
- ROC-AUC: 0.865
- Precision / Recall / F1: 0.714 / 0.769 / 0.741

## Drivers (why it works)
- + ΔKO% (strongest), + ΔResistance, + ΔAbility
- – ΔHas-Been-KO% (more past KOs hurts)
- Weight & rounds: small, situational; Power ~ tied in data

## Limitations
Small dataset; power often tied; partial metadata (age/stance/reach); aggregate (pre-fight) stats.

## How to use
Artifacts in `models/boxing_winner_v1/`.  
`predict_winner(opp1, opp2)` → probability + decision (see notebooks).
