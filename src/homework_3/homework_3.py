import math
import os
import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# PART 1 — Environment and Simulator
# =============================================================================

STATES       = [f'A{i}' for i in range(1, 6)] + [f'B{i}' for i in range(1, 6)]
NON_TERMINAL = ['A1', 'A2', 'A3', 'A4', 'A5', 'B2', 'B4']
TERMINAL     = ['B1', 'B3', 'B5']
ACTIONS      = ['up', 'down', 'left', 'right']
GAMMA        = 0.9
MAX_STEPS    = 200


def move(s, direction):
    """Deterministic move from state s in the given direction; hitting a wall -> same state."""
    row, col = s[0], int(s[1])
    if direction == 'up':
        return s if row == 'A' else f'A{col}'
    if direction == 'down':
        return s if row == 'B' else f'B{col}'
    if direction == 'left':
        return s if col == 1 else f'{row}{col - 1}'
    # right
    return s if col == 5 else f'{row}{col + 1}'


def transition_dist(s, a):
    """
    P(s'|s,a): the intended direction gets 0.6, each of the 3 remaining
    directions gets 0.1, staying in place gets 0.1.
    Wall bumps merge back into s (defaultdict sums them automatically).
    """
    P = defaultdict(float)
    for direction in ACTIONS:
        p = 0.6 if direction == a else 0.1
        P[move(s, direction)] += p
    P[s] += 0.1   # "stay in place" outcome
    return dict(P)


def reward(s_next):
    """R(s'): +1.0 for B3, -1.0 for B1/B5, -0.04 for every other state."""
    if s_next == 'B3':
        return 1.0
    if s_next in ('B1', 'B5'):
        return -1.0
    return -0.04


def _sample_from(P):
    states = list(P.keys())
    return random.choices(states, weights=[P[s] for s in states], k=1)[0]


class Simulator:
    def reset(self):
        """Uniformly picks one of the 7 non-terminal states."""
        self.s = random.choice(NON_TERMINAL)
        return self.s

    def step(self, a):
        """Apply action a; return (reward, new_state, done)."""
        P      = transition_dist(self.s, a)
        s_next = _sample_from(P)
        r      = reward(s_next)
        done   = s_next in TERMINAL
        self.s = s_next
        return r, s_next, done

    def model(self, s, a):
        """May be used ONLY by Q-value iteration (part 2)."""
        return transition_dist(s, a)


# =============================================================================
# PART 2 — Q-value iteration (benchmark)
# =============================================================================

def _argmax_a(Q, s):
    """argmax_a Q(s,a) with random tie-breaking."""
    best = max(Q[(s, a)] for a in ACTIONS)
    return random.choice([a for a in ACTIONS if Q[(s, a)] == best])


def q_value_iteration(gamma=GAMMA, tol=1e-12):
    """
    Synchronous Q-value iteration over the known model.
    Returns: Q, V_star, pi_star, V_hist (list of V snapshots per iteration).
    May use transition_dist (known model) — ONLY this part is allowed to!
    """
    Q      = {(s, a): 0.0 for s in NON_TERMINAL for a in ACTIONS}
    V_hist = []

    while True:
        Q_new  = {}
        delta  = 0.0
        for s in NON_TERMINAL:
            for a in ACTIONS:
                q = sum(
                    p * (reward(s2) + gamma * (
                        0.0 if s2 in TERMINAL
                        else max(Q[(s2, a2)] for a2 in ACTIONS)
                    ))
                    for s2, p in transition_dist(s, a).items()
                )
                Q_new[(s, a)] = q
                delta = max(delta, abs(q - Q[(s, a)]))
        Q = Q_new
        V_hist.append({s: max(Q[(s, a)] for a in ACTIONS) for s in NON_TERMINAL})
        if delta < tol:
            break

    V_star  = {s: max(Q[(s, a)] for a in ACTIONS) for s in NON_TERMINAL}
    pi_star = {s: _argmax_a(Q, s) for s in NON_TERMINAL}
    return Q, V_star, pi_star, V_hist


# =============================================================================
# PART 3 — Q-learning (model-free, TD, epsilon-greedy)
# =============================================================================

def _epsilon_greedy(Q, s, eps):
    """Epsilon-greedy action selection; random tie-breaking."""
    if random.random() < eps:
        return random.choice(ACTIONS)
    best = max(Q[(s, a)] for a in ACTIONS)
    return random.choice([a for a in ACTIONS if Q[(s, a)] == best])


def q_learning(sim, n_ep, gamma=GAMMA, eps=0.1, alpha_fn=None, snapshot_every=50):
    """
    Q-learning with epsilon-greedy exploration.
    alpha_fn(e) -> learning rate for episode e.
    Returns: Q, V_snaps (list of {s: V_t} every snapshot_every ep.), ep_rewards.
    Must NOT use sim.model() — only reset() and step().
    """
    if alpha_fn is None:
        def alpha_fn(e):
            return math.log(e + 1) / (e + 1)

    Q          = defaultdict(float)
    V_snaps    = []
    ep_rewards = []

    for e in range(1, n_ep + 1):
        s     = sim.reset()
        done  = False
        G     = 0.0
        steps = 0
        alpha = alpha_fn(e)

        while not done and steps < MAX_STEPS:
            a           = _epsilon_greedy(Q, s, eps)
            r, s2, done = sim.step(a)
            target      = r if done else r + gamma * max(Q[(s2, a2)] for a2 in ACTIONS)
            Q[(s, a)]  += alpha * (target - Q[(s, a)])
            s, G, steps = s2, G + r, steps + 1

        ep_rewards.append(G)
        if e % snapshot_every == 0:
            V_snaps.append({s: max(Q[(s, a)] for a in ACTIONS) for s in NON_TERMINAL})

    return dict(Q), V_snaps, ep_rewards


def test_policy(sim, Q, n_test=10):
    """Evaluate the learned policy: n_test episodes, eps=0, no TD updates.
    Returns the average cumulative raw reward per episode."""
    total = 0.0
    for _ in range(n_test):
        s     = sim.reset()
        done  = False
        G     = 0.0
        steps = 0
        while not done and steps < MAX_STEPS:
            a           = _argmax_a(Q, s)
            r, s, done  = sim.step(a)
            G          += r
            steps      += 1
        total += G
    return total / n_test


# =============================================================================
# PART 4 — REINFORCE (policy gradient, softmax parametrization)
# =============================================================================

def _softmax(logits):
    """Numerically stable softmax: subtracts the max to prevent overflow."""
    m    = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s    = sum(exps)
    return [e / s for e in exps]


def reinforce(sim, n_ep, gamma=GAMMA, alpha_fn=None, snapshot_every=200):
    """
    REINFORCE with a softmax policy.
    Parametrization: theta[s,a] for every non-terminal s and action a (7x4 = 28 params).
    pi_theta(a|s) = softmax_a(theta[s,.])
    Update: theta[s,a'] += alpha . v_tau . (1[a'=a] - pi(a'|s))  for all a'
    Must NOT use sim.model() — only reset() and step().
    """
    if alpha_fn is None:
        def alpha_fn(e):
            return math.log(e + 1) / (e + 1)

    theta      = {(s, a): 0.0 for s in NON_TERMINAL for a in ACTIONS}
    ep_rewards = []
    theta_hist = []

    for e in range(1, n_ep + 1):
        # --- 1) Generate an episode following the current policy ---
        traj  = []
        s     = sim.reset()
        done  = False
        steps = 0

        while not done and steps < MAX_STEPS:
            probs      = _softmax([theta[(s, a)] for a in ACTIONS])
            a          = random.choices(ACTIONS, weights=probs, k=1)[0]
            r, s2, done = sim.step(a)
            traj.append((s, a, r))
            s, steps = s2, steps + 1

        # --- 2) Return-to-go, computed backwards: v_tau = r_tau + gamma*v_{tau+1} ---
        returns = [0.0] * len(traj)
        v       = 0.0
        for t in reversed(range(len(traj))):
            v          = traj[t][2] + gamma * v
            returns[t] = v

        # --- 3) Parameter update ---
        alpha = alpha_fn(e)
        for t, (st, at, _) in enumerate(traj):
            probs = _softmax([theta[(st, a)] for a in ACTIONS])
            for j, aj in enumerate(ACTIONS):
                # Score function: d/d theta[st,aj] ln pi = 1[aj=at] - pi(aj|st)
                grad = (1.0 if aj == at else 0.0) - probs[j]
                theta[(st, aj)] += alpha * returns[t] * grad

        ep_rewards.append(sum(r for _, _, r in traj))
        if e % snapshot_every == 0:
            theta_hist.append(dict(theta))

    return theta, ep_rewards, theta_hist


def _policy_from_theta(theta):
    """Greedy policy from theta: argmax_a pi_theta(a|s) for every non-terminal s."""
    pi = {}
    for s in NON_TERMINAL:
        probs = _softmax([theta[(s, a)] for a in ACTIONS])
        pi[s] = ACTIONS[probs.index(max(probs))]
    return pi


# =============================================================================
# PLOTS (G1-G11)
# =============================================================================

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')

# Global style — serif font, no top/right spines, dotted grid
plt.rcParams.update({
    'font.family':        'DejaVu Serif',
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'legend.framealpha':  0.85,
    'axes.grid':          True,
    'grid.alpha':         0.38,
    'grid.linestyle':     ':',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.facecolor':   'white',
    'axes.facecolor':     '#f8f8f8',
})

# Line palette (8 colors, perceptually separated, darker than tab10)
_PAL8 = ['#1b4f72', '#c0392b', '#1e8449', '#884ea0',
          '#d68910', '#148f77', '#7f8c8d', '#2e86c1']

# Colors per action for G8 — teal/red/green/purple
_ACT_CLR = {'up': '#1a6b8a', 'down': '#c0392b',
             'left': '#27ae60', 'right': '#8e44ad'}


def _ensure_out():
    os.makedirs(_OUT, exist_ok=True)


def _draw_grid(ax, V, pi, title, v_min=-1.0, v_max=1.0):
    """2x5 grid: color = V value (RdBu_r), symbol = policy."""
    arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    term_v = {'B1': -1.0, 'B3': +1.0, 'B5': -1.0}
    cmap   = plt.cm.RdBu_r   # red=-1, white=0, blue=+1

    for ri, row in enumerate(['A', 'B']):
        for ci in range(5):
            s   = f'{row}{ci + 1}'
            y   = 1 - ri
            val = term_v.get(s, V.get(s, 0.0))
            norm = np.clip((val - v_min) / (v_max - v_min), 0.0, 1.0)
            clr  = cmap(norm)
            # text black on light cells, white on dark cells
            txt_clr = 'white' if (norm < 0.25 or norm > 0.75) else '#1a1a1a'

            ax.add_patch(plt.Rectangle((ci, y), 1, 1,
                                       facecolor=clr, edgecolor='#555', lw=1.2))
            # state name — small tag in the top-left corner
            ax.text(ci + 0.08, y + 0.88, s,
                    ha='left', va='top', fontsize=7, color=txt_clr, alpha=0.7)
            # value in the center
            ax.text(ci + 0.5, y + 0.60, f'{val:+.3f}',
                    ha='center', va='center', fontsize=9,
                    fontweight='bold', color=txt_clr)
            if s in TERMINAL:
                ax.text(ci + 0.5, y + 0.28, 'TERM',
                        ha='center', va='center', fontsize=8,
                        color=txt_clr, style='italic')
            elif pi:
                ax.text(ci + 0.5, y + 0.25, arrows[pi[s]],
                        ha='center', va='center', fontsize=22, color=txt_clr)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2)
    ax.set_xticks([i + 0.5 for i in range(5)])
    ax.set_xticklabels([f'col{i + 1}' for i in range(5)], fontsize=8)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['row B', 'row A'], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_facecolor('white')   # grid cells on white background


def _rolling_mean(arr, window):
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def _ql_error_curve(V_star, n_seeds, n_ep, gamma, eps, alpha_fn, snap_every):
    """Return an [n_seeds x n_snaps] array of max-norm errors ||V_t - V*||inf."""
    sim     = Simulator()
    n_snaps = n_ep // snap_every
    errs    = np.zeros((n_seeds, n_snaps))
    for seed in range(n_seeds):
        random.seed(seed)
        _, snaps, _ = q_learning(sim, n_ep, gamma, eps, alpha_fn, snap_every)
        for t, snap in enumerate(snaps):
            errs[seed, t] = max(abs(snap[s] - V_star[s]) for s in NON_TERMINAL)
    return errs


# -- G1 ------------------------------------------------------------------------

def make_g1(V_star, pi_star):
    """G1: V* heatmap + policy arrows on the 2x5 grid."""
    _ensure_out()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    _draw_grid(ax, V_star, pi_star, 'Benchmark: optimal value function V*(s) and policy pi*(s)  [gamma=0.9]')
    plt.tight_layout()
    path = os.path.join(_OUT, 'G1_Vstar_pistar.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# -- G2 ------------------------------------------------------------------------

def make_g2(V_hist, V_star):
    """G2: Q-value iteration convergence — V_t(s) per iteration."""
    _ensure_out()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, s in enumerate(NON_TERMINAL):
        col = _PAL8[i % len(_PAL8)]
        ys  = [snap[s] for snap in V_hist]
        ax.plot(ys, color=col, label=s, linewidth=2.0)
        ax.axhline(V_star[s], color=col, linestyle='--', linewidth=1.0, alpha=0.45)
    ax.set_xlabel('DP iteration')
    ax.set_ylabel('$V_t(s)$')
    ax.set_title('Q-value iteration convergence per state  (gamma = 0.9)\n'
                 'Solid lines: $V_t(s)$        Dashed: $V^*(s)$')
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    path = os.path.join(_OUT, 'G2_QVI_konvergencija.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# -- G3 ------------------------------------------------------------------------

def make_g3(V_snaps, V_star, snap_every, gamma=0.9, eps=0.1):
    """G3: V_t(s) -> V*(s) for all 7 states — the main required plot."""
    _ensure_out()
    x         = np.arange(1, len(V_snaps) + 1) * snap_every
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
    axes = axes.flatten()
    for i, s in enumerate(NON_TERMINAL):
        col = _PAL8[i % len(_PAL8)]
        ax  = axes[i]
        ax.plot(x, [snap[s] for snap in V_snaps],
                color=col, linewidth=2.0, label=f'$V_t$({s})')
        ax.axhline(V_star[s], color='#333333', linestyle='--',
                   linewidth=1.4, label=f'$V^*$ = {V_star[s]:.3f}')
        ax.set_title(f'State  {s}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('V', fontsize=8)
        ax.legend(fontsize=7.5)
    axes[7].axis('off')
    axes[7].text(0.5, 0.55,
                 f'Q-learning\ngamma = {gamma},  eps = {eps}\n'
                 f'Dashed = $V^*$ (DP benchmark)',
                 ha='center', va='center', fontsize=10,
                 transform=axes[7].transAxes,
                 bbox=dict(boxstyle='round,pad=0.6',
                           facecolor='#eaf4fb', edgecolor='#2e86c1',
                           linewidth=1.2, alpha=0.9))
    fig.suptitle('Q-learning: convergence of $V_t(s)$ to the benchmark $V^*(s)$ per state',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(_OUT, 'G3_QL_Vt_po_stanjima.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# -- G4 ------------------------------------------------------------------------

def make_g4(V_star, n_seeds=20, n_ep=10_000, snap_every=100, gamma=0.9, eps=0.1):
    """G4: learning-rate comparison — ||V_t - V*|| per episode."""
    _ensure_out()

    def af_var(e):   return math.log(e + 1) / (e + 1)
    def af_005(_e):  return 0.05
    def af_010(_e):  return 0.10
    def af_020(_e):  return 0.20
    def af_050(_e):  return 0.50

    linestyles = ['-', '--', '-.', ':', '-']
    linewidths = [2.4, 1.9, 1.9, 1.9, 1.5]
    schemes = [('log(e+1)/(e+1)', af_var), ('alpha=0.05', af_005),
               ('alpha=0.10', af_010), ('alpha=0.20', af_020), ('alpha=0.50', af_050)]
    x = np.arange(1, n_ep // snap_every + 1) * snap_every

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for idx, ((name, alpha_fn), ls, lw) in enumerate(
            zip(schemes, linestyles, linewidths)):
        col = _PAL8[idx % len(_PAL8)]
        print(f'    G4: {name} ({n_seeds} seeds)...')
        errs = _ql_error_curve(V_star, n_seeds, n_ep, gamma, eps, alpha_fn, snap_every)
        m, s = errs.mean(0), errs.std(0)
        ax.plot(x, m, color=col, label=name, linewidth=lw, linestyle=ls)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.12)

    ax.set_yscale('log')   # log scale — shows the plateau differences better
    ax.set_xlabel('Episode')
    ax.set_ylabel('$\\|V_t - V^*\\|_\\infty$  (log scale)')
    ax.set_title(f'Q-learning convergence speed for different learning rates alpha\n'
                 f'(gamma={gamma}, eps={eps}, {n_seeds} runs, shading = +/- std)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(_OUT, 'G4_QL_alpha_poredjenje.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# -- G5 ------------------------------------------------------------------------

def make_g5(V_star, n_seeds=20, n_ep=10_000, snap_every=100, gamma=0.9):
    """G5: epsilon comparison — ||V_t - V*|| per episode."""
    _ensure_out()

    def af_var(e):  return math.log(e + 1) / (e + 1)

    epsilons   = [0.0, 0.05, 0.1, 0.2, 0.3]
    linestyles = [':', '-.', '--', '-', '-']
    linewidths = [1.8, 1.8, 2.0, 2.0, 2.0]
    x          = np.arange(1, n_ep // snap_every + 1) * snap_every

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for idx, (eps, ls, lw) in enumerate(zip(epsilons, linestyles, linewidths)):
        col = _PAL8[idx % len(_PAL8)]
        print(f'    G5: eps={eps} ({n_seeds} seeds)...')
        errs = _ql_error_curve(V_star, n_seeds, n_ep, gamma, eps, af_var, snap_every)
        m, s = errs.mean(0), errs.std(0)
        ax.plot(x, m, color=col, label=f'eps = {eps}', linewidth=lw, linestyle=ls)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.12)

    ax.set_yscale('log')
    ax.set_xlabel('Episode')
    ax.set_ylabel('$\\|V_t - V^*\\|_\\infty$  (log scale)')
    ax.set_title(f'Effect of the exploration parameter eps on Q-learning convergence\n'
                 f'(gamma={gamma}, alpha=log(e+1)/(e+1), {n_seeds} runs)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(_OUT, 'G5_QL_epsilon_poredjenje.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# -- G6 ------------------------------------------------------------------------

def make_g6(V09, pi09, avg09, V999, pi999, avg999):
    """G6: learned policies for gamma=0.9 and gamma=0.999."""
    _ensure_out()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    _draw_grid(axes[0], V09,  pi09,
               f'Q-learning gamma=0.9  (avg/1000 ep = {avg09:+.3f})')
    _draw_grid(axes[1], V999, pi999,
               f'Q-learning gamma=0.999  (avg/1000 ep = {avg999:+.3f})')
    fig.suptitle('Q-learning policies learned for different discount factors',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(_OUT, 'G6_QL_gamma_poredjenje.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# -- G7 ------------------------------------------------------------------------

def make_g7(ep_rewards, window=300):
    """G7: REINFORCE — raw reward per episode + rolling mean."""
    _ensure_out()
    x  = np.arange(1, len(ep_rewards) + 1)
    rm = _rolling_mean(np.array(ep_rewards, dtype=float), window)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, ep_rewards, alpha=0.08, color='#7fb3d3',
               s=2, label='Reward (raw)', rasterized=True)
    ax.plot(np.arange(window, len(ep_rewards) + 1), rm,
            color='#c0392b', linewidth=2.2,
            label=f'Rolling mean (window = {window})')
    ax.axhline(0, color='#555', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total (raw) reward')
    ax.set_title('REINFORCE: total reward per episode and rolling mean  (gamma = 0.9)',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(_OUT, 'G7_RF_kriva_ucenja.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# -- G8 ------------------------------------------------------------------------

def make_g8(theta_hist, snap_every):
    """G8: REINFORCE — theta parameters and pi probabilities per state."""
    _ensure_out()
    x    = np.arange(1, len(theta_hist) + 1) * snap_every
    clrs = _ACT_CLR   # teal/red/green/purple

    for fname, ylabel, val_fn, extra_fn, suptitle in [
        ('G8a_RF_theta.png', 'theta[s,a]',
         lambda snap, s, a: snap[(s, a)],
         lambda ax: None,
         'G8a — REINFORCE: parameters theta[s,a] during training'),
        ('G8b_RF_pi.png', 'pi(a|s)',
         lambda snap, s, a: _softmax([snap[(s, aa)] for aa in ACTIONS])[ACTIONS.index(a)],
         lambda ax: ax.axhline(0.25, color='gray', linestyle=':', linewidth=0.9),
         'G8b — REINFORCE: action probabilities pi(a|s) during training'),
    ]:
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        axes = axes.flatten()
        for i, s in enumerate(NON_TERMINAL):
            ax = axes[i]
            for a in ACTIONS:
                ax.plot(x, [val_fn(snap, s, a) for snap in theta_hist],
                        label=a, color=clrs[a], linewidth=1.5)
            extra_fn(ax)
            if 'pi' in fname:
                ax.set_ylim(0, 1)
            ax.set_title(f'State {s}', fontsize=10)
            ax.set_xlabel('Episode', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=0.3)
        axes[7].axis('off')
        fig.suptitle(suptitle, fontsize=11, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(_OUT, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Saved: {path}')


# -- G9 ------------------------------------------------------------------------

def make_g9(n_seeds=10, n_ep=20_000, gamma=0.9, window=300):
    """G9: REINFORCE — learning-rate comparison (reward rolling mean)."""
    _ensure_out()

    def af_var(e):  return math.log(e + 1) / (e + 1)
    def af_001(_e): return 0.001
    def af_01(_e):  return 0.01
    def af_05(_e):  return 0.05

    linestyles = ['-', '--', '-.', ':']
    linewidths = [2.3, 1.9, 1.9, 1.9]
    schemes    = [('log(e+1)/(e+1)', af_var), ('alpha=0.001', af_001),
                  ('alpha=0.01', af_01), ('alpha=0.05', af_05)]
    sim = Simulator()

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for idx, ((name, alpha_fn), ls, lw) in enumerate(
            zip(schemes, linestyles, linewidths)):
        col = _PAL8[idx % len(_PAL8)]
        print(f'    G9: {name} ({n_seeds} seeds)...')
        all_rw = []
        for seed in range(n_seeds):
            random.seed(seed)
            _, rw, _ = reinforce(sim, n_ep, gamma, alpha_fn, snapshot_every=n_ep + 1)
            all_rw.append(rw)
        rm_all = np.array([_rolling_mean(np.array(rw, dtype=float), window)
                           for rw in all_rw])
        m, sd  = rm_all.mean(0), rm_all.std(0)
        x_rm   = np.arange(window, n_ep + 1)
        ax.plot(x_rm, m, color=col, label=name, linewidth=lw, linestyle=ls)
        ax.fill_between(x_rm, m - sd, m + sd, color=col, alpha=0.12)

    ax.axhline(0, color='#888', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Reward rolling mean  (window = {window})')
    ax.set_title(f'REINFORCE: effect of the learning rate alpha on convergence speed\n'
                 f'(gamma = {gamma}, {n_seeds} runs, shading = +/- std)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(_OUT, 'G9_RF_alpha_poredjenje.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# -- G10 -------------------------------------------------------------------------

def make_g10(theta):
    """G10: REINFORCE — learned policy on the 2x5 grid."""
    _ensure_out()
    pi_rf = _policy_from_theta(theta)
    V_rf  = {s: max(_softmax([theta[(s, a)] for a in ACTIONS]))
             for s in NON_TERMINAL}
    fig, ax = plt.subplots(figsize=(9, 3.5))
    _draw_grid(ax, V_rf, pi_rf,
               'REINFORCE: learned policy  (cell value = max pi(a|s))',
               v_min=0.25, v_max=1.0)
    plt.tight_layout()
    path = os.path.join(_OUT, 'G10_RF_politika.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# -- G11 -------------------------------------------------------------------------

def make_g11(V_star, pi_star, Q_ql, theta):
    """G11: summary comparison of DP / Q-learning / REINFORCE."""
    _ensure_out()
    pi_ql = {s: _argmax_a(Q_ql, s) for s in NON_TERMINAL}
    V_ql  = {s: max(Q_ql.get((s, a), 0.0) for a in ACTIONS) for s in NON_TERMINAL}
    pi_rf = _policy_from_theta(theta)
    V_rf  = {s: max(_softmax([theta[(s, a)] for a in ACTIONS]))
             for s in NON_TERMINAL}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    _draw_grid(axes[0], V_star, pi_star,
               'DP — Q-value iteration\n(V*, pi* — benchmark)')
    _draw_grid(axes[1], V_ql,   pi_ql,
               'Q-learning\n(gamma=0.9, alpha=log/(e+1), 10,000 ep.)')
    _draw_grid(axes[2], V_rf,   pi_rf,
               'REINFORCE\n(gamma=0.9, alpha=log/(e+1), 30,000 ep.)',
               v_min=0.25, v_max=1.0)
    fig.suptitle('Summary comparison of learned policies: DP benchmark / Q-learning / REINFORCE',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(_OUT, 'G11_sumarno.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# =============================================================================
# Run experiments and generate plots  (python homework_3.py)
# =============================================================================

if __name__ == '__main__':
    # -- Part 1: verify the transition model -----------------------------------
    dist = transition_dist('A1', 'right')
    ok   = all(abs(dist.get(k, 0) - v) < 1e-9
               for k, v in {'A2': 0.6, 'A1': 0.3, 'B1': 0.1}.items())
    print(f"[PART 1] transition verification A1/right: {ok}  {dist}")

    # -- Part 2: Q-value iteration ----------------------------------------------
    print("\n[PART 2] Q-value iteration ...")
    Q, V_star, pi_star, V_hist = q_value_iteration(gamma=0.9)
    print(f"  gamma=0.9  -> converged after {len(V_hist)} iterations")
    print(f"  V*: { {s: round(V_star[s], 3) for s in NON_TERMINAL} }")

    _, V_star_999, pi_star_999, _ = q_value_iteration(gamma=0.999)

    # -- Part 3: Q-learning -------------------------------------------------------
    random.seed(0)
    sim = Simulator()

    def alpha_var(e):
        return math.log(e + 1) / (e + 1)

    def alpha_const(_e):
        return 0.1

    print("\n[PART 3] Q-learning (gamma=0.9, eps=0.1, alpha=log/(e+1), 10,000 ep.) ...")
    Q_ql, V_snaps_var, _ = q_learning(
        sim, n_ep=10_000, gamma=0.9, eps=0.1, alpha_fn=alpha_var, snapshot_every=100
    )
    pi_ql = {s: _argmax_a(Q_ql, s) for s in NON_TERMINAL}
    V_ql  = {s: max(Q_ql[(s, a)] for a in ACTIONS) for s in NON_TERMINAL}
    avg10 = test_policy(sim, Q_ql, n_test=10)
    avg1k = test_policy(sim, Q_ql, n_test=1000)
    print(f"  Error |V_t-V*|inf (last snapshot): "
          f"{ {s: round(abs(V_snaps_var[-1][s]-V_star[s]),3) for s in NON_TERMINAL} }")
    print(f"  Greedy test (gamma=0.9): 10 ep={avg10:+.3f},  1000 ep={avg1k:+.3f}")

    print("\n[PART 3] Q-learning (gamma=0.9, alpha=0.1 const.) ...")
    Q_const, V_snaps_const, _ = q_learning(
        sim, n_ep=10_000, gamma=0.9, eps=0.1, alpha_fn=alpha_const, snapshot_every=100
    )

    print("\n[PART 3] Q-learning (gamma=0.999, alpha=log/(e+1)) ...")
    Q_ql999, _, _ = q_learning(
        sim, n_ep=10_000, gamma=0.999, eps=0.1, alpha_fn=alpha_var, snapshot_every=100
    )
    pi_ql999  = {s: _argmax_a(Q_ql999, s) for s in NON_TERMINAL}
    V_ql999   = {s: max(Q_ql999[(s, a)] for a in ACTIONS) for s in NON_TERMINAL}
    avg10_999 = test_policy(sim, Q_ql999, n_test=10)
    avg1k_999 = test_policy(sim, Q_ql999, n_test=1000)
    print(f"  Greedy test (gamma=0.999): 10 ep={avg10_999:+.3f},  1000 ep={avg1k_999:+.3f}")

    # -- Part 4: REINFORCE --------------------------------------------------------
    print("\n[PART 4] REINFORCE (gamma=0.9, alpha=log/(e+1), 30,000 ep.) ...")
    theta, ep_rewards_rf, theta_hist = reinforce(
        sim, n_ep=30_000, gamma=0.9, alpha_fn=alpha_var, snapshot_every=200
    )
    pi_rf    = _policy_from_theta(theta)
    match_ok = sum(pi_rf[s] == pi_star[s] for s in NON_TERMINAL)
    avg_first = sum(ep_rewards_rf[:1000])  / 1000
    avg_last  = sum(ep_rewards_rf[-1000:]) / 1000
    print(f"  Matches with pi*: {match_ok}/7")
    print(f"  Average reward: first 1000 ep={avg_first:+.3f}, last 1000 ep={avg_last:+.3f}")

    # =============================================================================
    # Generate plots G1-G11
    # =============================================================================
    print('\n=== Generating plots ===')

    make_g1(V_star, pi_star)
    make_g2(V_hist, V_star)
    make_g3(V_snaps_var, V_star, snap_every=100)

    print('  G4: alpha comparison (multi-seed, may take ~1 min)...')
    make_g4(V_star)

    print('  G5: epsilon comparison (multi-seed, may take ~1 min)...')
    make_g5(V_star)

    make_g6(V_ql, pi_ql, avg1k, V_ql999, pi_ql999, avg1k_999)
    make_g7(ep_rewards_rf)
    make_g8(theta_hist, snap_every=200)

    print('  G9: REINFORCE alpha (multi-seed, may take ~2 min)...')
    make_g9()

    make_g10(theta)
    make_g11(V_star, pi_star, Q_ql, theta)

    print(f'\nAll plots saved to: {_OUT}')
