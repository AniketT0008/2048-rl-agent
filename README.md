# 2048 Reinforcement Learning Agent

A score-focused agent for classic 4×4 2048. The project combines an N-tuple temporal-difference value model with two-ply expectimax, a rules-accurate simulator, evaluation tools, and a local browser interface.

## Results

The included checkpoint was evaluated on 200 fresh games with two-ply expectimax. It reached 2048 or higher in **198 games (99%)**, with a mean score of **162,771**. The best game in that batch reached **16,384** and scored **309,568**. The checkpoint was selected by mean score; results vary with random tile spawns.

## Run locally

Python 3.11 is recommended. In PowerShell from this repository:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py --checkpoint models\ntuple_best.npz --search-depth 2
```

Open `http://127.0.0.1:8000`. Use arrow keys or WASD to play, **AI move** for one action, or **Watch AI play** for autoplay. The game can continue beyond 2048 to pursue higher tiles and scores.

## Evaluate the checkpoint

```powershell
python evaluate_ntuple.py --checkpoint models\ntuple_best.npz --games 200 --search-depth 2
```

The evaluator reports win rate, mean and median score, mean maximum tile, and the best tile and score in the run.

## Continue score training

```powershell
python train_ntuple.py --episodes 50000 --eval-games 50 --eval-every 25000 --objective score --search-depth 2 --resume models\ntuple_best.npz --output-dir runs\score
```

The score objective retains the checkpoint with the highest mean evaluation score and uses maximum game score as a tie-breaker. Checkpoints and episode metrics are written under the selected output directory.

## Model details

The value function uses four six-cell board patterns under eight rotations/reflections, sharing four float32 tables of `16^6` entries each. The learned weights occupy 256 MiB uncompressed; `models/ntuple_best.npz` is compressed. Two-ply expectimax evaluates random 2/4 tile spawns before choosing a move.

The 2048 game is stochastic, so a measured win rate is not a guarantee for every game. N-tuple temporal-difference value methods have been studied specifically for 2048; see the [Journal of Information Processing study](https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article).
