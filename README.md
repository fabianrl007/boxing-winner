# boxing-winner
Predict boxing match winners with Δ-features + Logistic Regression.
## Quick Start (predict without retraining)

Install deps:
```bash
pip install -r requirements.txt
```

Use the packaged model:
```python
from src.predict import predict_winner

opp1 = {"round_ko_percentage": 25.6, "estimated_punch_resistance": 68,
        "estimated_ability_to_take_punch": 72, "avg_weight": 160.0,
        "estimated_punch_power": 82, "rounds_boxed": 200,
        "has_been_ko_percentage": 5.0}

opp2 = {"round_ko_percentage": 13.3, "estimated_punch_resistance": 62,
        "estimated_ability_to_take_punch": 68, "avg_weight": 164.3,
        "estimated_punch_power": 82, "rounds_boxed": 119,
        "has_been_ko_percentage": 5.0}

predict_winner(opp1, opp2)
```

**Colab tip:** after cloning, add the repo to `sys.path`:
```python
import sys; sys.path.append('/content/boxing-winner')
```
