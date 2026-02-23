# Card-Jitsu Game Theory Optimal (GTO) Solver

A rigorous mathematical engine that reverse-engineers the Club Penguin minigame 'Card-Jitsu' into a complex model of imperfect information, simultaneous decision-making, and dynamic probability. 

This project proves that beneath the nostalgic surface of a children's flash game lies a mathematical environment as complex as modern poker, featuring depleting drafted decks, state-dependent constraints, and mid-turn matrix warping.

## Core Architectures

This engine features two distinct AI systems, allowing you to test human psychology against cold, hard game theory.

### 1. The Exploitative Bot (Bayesian Inference)
Built for interactive play against humans. This bot tracks your tendencies using a Level-K reasoning model.
* **Combinatorial Equity:** It evaluates the board state by calculating "True Outs" (winning cards) across both Immediate (Phase 1) and Discounted Future (Phase 2) draws.
* **Bayesian Updating:** It applies Bayes' Theorem after every round to dynamically update its belief about the human's psychological logic level (Random vs. Self-Focused vs. Defensive), creating a predictive probability vector to exploit human habits.
* **Lookahead Expected Value (EV):** It anticipates "Power Cards" to calculate the EV of every possible branch, prioritizing moves that protect its own equity or destroy the opponent's.

### 2. The GTO Bot (Counterfactual Regret Minimization)
Built to be mathematically unexploitable. If an exploitative bot plays a smart opponent, it becomes predictable. To solve this, I implemented a CFR algorithm.
* **Parallel Timelines:** The bot plays millions of simultaneous rounds against itself.
* **Regret Matching:** After every hypothetical game state, it calculates its "Regret" (the EV of a specific card minus the EV of its overall mixed strategy).
* **Nash Equilibrium:** Over millions of iterations, the bot's strategy swings like a pendulum, eventually settling on a perfect percentage distribution (Nash Equilibrium) that makes the opponent mathematically indifferent to their choices.

## Features: Power Cards
Standard Rock-Paper-Scissors uses a static $3 \times 3$ zero-sum payoff matrix. Card-Jitsu introduces "Power Cards," which this engine simulates flawlessly:
* **Immediate Matrix Overwrites:** Element shifts (e.g., Fire becomes Snow) and `LOW_WINS` dynamically warp the fundamental RPS payoff matrix *mid-reveal*.
* **State & Equity Destruction:** Mass deck destructions and tableau discards reach into the object trackers to literally delete achieved equity.
* **Next-Turn Constraints:** Element blocking mathematically reduces a player's probability of playing a specific element $P(X)$ to exactly $0$ for future turns.

## The Output: Weaponising Weakness
When running the CFR matrix training, the AI learns advanced meta-strategies without any hardcoded rules. 

In a scenario where Player 1 holds a `10 FIRE [LOW_WINS]` card against Player 2's greater `11 WATER` and `12 SNOW` cards, human intuition says not to play the `10`. The CFR supercomputer calculates otherwise:

```text
=========================================
🧠 CFR MATRIX TRAINING (10000 ITERATIONS) 🧠
=========================================
... 2500 iterations ...
... 5000 iterations ...
... 7500 iterations ...

--- PLAYER 1 GTO STRATEGY ---
> 10 FIRE RED [LOW_WINS]: 33.21%
> 4 WATER BLUE: 33.83%
> 7 SNOW GREEN: 32.96%

--- PLAYER 2 GTO STRATEGY ---
> 3 FIRE YELLOW: 33.82%
> 11 WATER ORANGE: 33.58%
> 12 SNOW PURPLE: 32.60%
