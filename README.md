# Algorithms for Artificial Intelligence
### University of Belgrade — School of Electrical Engineering (ETF)
**Author:** Mihajlo Stevanović (2022/0315)

---

A growing collection of AI algorithm implementations developed as coursework at ETF Belgrade. Each homework covers a distinct area of artificial intelligence — from classical search and optimization to probabilistic reasoning and reinforcement learning.

All implementations are written in **Python** and include experimental analysis, convergence plots, and Monte Carlo validation where applicable.

---

## Homeworks

| # | Course | Topic | Algorithms | Status |
|---|---|---|---|---|
| [HW1](src/homework_1/) | 13E054VI — Artificial Intelligence | Local Search | Simulated Annealing · Genetic Algorithm · PSO | ✅ Done |
| [HW2](src/homework_2/) | 13E054VI — Artificial Intelligence | Bayesian Networks & Particle Filters | Variable Elimination · Rejection Sampling · Gibbs Sampling · Particle Filter | ✅ Done |
| [HW3](src/homework_3/) | 13E053VI — Reinforcement Learning | Reinforcement Learning | Q-Value Iteration · Q-Learning · REINFORCE | ✅ Done |

Each entry links to a homework folder with its own `README.md` covering the problem definition, methods, and results in detail.

---

## Repository Structure

```
.
└── src/
    ├── homework_1/        # Local search: SA, GA, PSO
    ├── homework_2/        # Bayesian network inference & particle filter tracking
    └── homework_3/        # Q-value iteration, Q-learning, REINFORCE
```

---

## Requirements

```
Python 3.x
numpy
matplotlib
scipy
networkx
```

Each homework folder's `README.md` lists only the dependencies it actually needs.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

*School of Electrical Engineering, University of Belgrade — 2025/2026*
