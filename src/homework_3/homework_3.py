import math
import os
import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# DEO 1 — Okruženje i Simulator
# =============================================================================

STATES       = [f'A{i}' for i in range(1, 6)] + [f'B{i}' for i in range(1, 6)]
NON_TERMINAL = ['A1', 'A2', 'A3', 'A4', 'A5', 'B2', 'B4']
TERMINAL     = ['B1', 'B3', 'B5']
ACTIONS      = ['gore', 'dole', 'levo', 'desno']
GAMMA        = 0.9
MAX_STEPS    = 200


def move(s, smer):
    """Deterministički pomeraj iz stanja s u smeru smer; udarac u zid → isto stanje."""
    row, col = s[0], int(s[1])
    if smer == 'gore':
        return s if row == 'A' else f'A{col}'
    if smer == 'dole':
        return s if row == 'B' else f'B{col}'
    if smer == 'levo':
        return s if col == 1 else f'{row}{col - 1}'
    # desno
    return s if col == 5 else f'{row}{col + 1}'


def transition_dist(s, a):
    """
    P(s'|s,a): nameravani smer 0.6, svaki od 3 preostala smera 0.1, ostani 0.1.
    Udarci u zid se stapaju u s (defaultdict ih automatski sabira).
    """
    P = defaultdict(float)
    for smer in ACTIONS:
        p = 0.6 if smer == a else 0.1
        P[move(s, smer)] += p
    P[s] += 0.1   # ishod "ostani u mestu"
    return dict(P)


def reward(s_next):
    """R(s'): +1.0 za B3, -1.0 za B1/B5, -0.04 za sva ostala stanja."""
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
        """Uniformno bira jedno od 7 neterminalnih stanja."""
        self.s = random.choice(NON_TERMINAL)
        return self.s

    def step(self, a):
        """Primeni akciju a; vrati (nagrada, novo_stanje, done)."""
        P      = transition_dist(self.s, a)
        s_next = _sample_from(P)
        r      = reward(s_next)
        done   = s_next in TERMINAL
        self.s = s_next
        return r, s_next, done

    def model(self, s, a):
        """Sme da koristi SAMO iteracija Q-vrednosti (deo 2)."""
        return transition_dist(s, a)


# =============================================================================
# DEO 2 — Iteracija Q-vrednosti (etalon)
# =============================================================================

def _argmax_a(Q, s):
    """argmax_a Q(s,a) sa nasumičnim razbijanjem izjednačenja."""
    best = max(Q[(s, a)] for a in ACTIONS)
    return random.choice([a for a in ACTIONS if Q[(s, a)] == best])


def q_value_iteration(gamma=GAMMA, tol=1e-12):
    """
    Sinhrona iteracija Q-vrednosti nad poznatim modelom.
    Vraća: Q, V_star, pi_star, V_hist (lista snimaka V po iteraciji).
    Sme da koristi transition_dist (poznat model) — SAMO ovaj deo!
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
# DEO 3 — Q-učenje (model-free, TD, ε-gramzivo)
# =============================================================================

def _epsilon_greedy(Q, s, eps):
    """ε-gramziva selekcija akcije; nasumično razbijanje izjednačenja."""
    if random.random() < eps:
        return random.choice(ACTIONS)
    best = max(Q[(s, a)] for a in ACTIONS)
    return random.choice([a for a in ACTIONS if Q[(s, a)] == best])


def q_learning(sim, n_ep, gamma=GAMMA, eps=0.1, alpha_fn=None, snapshot_every=50):
    """
    Q-učenje sa ε-gramzivim istraživanjem.
    alpha_fn(e) -> stopa učenja za epizodu e.
    Vraća: Q, V_snaps (lista {s: V_t} svakih snapshot_every ep.), ep_rewards.
    NE sme da koristi sim.model() — samo reset() i step().
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
    """Proceni naučenu politiku: n_test epizoda, ε=0, bez TD ažuriranja.
    Vraća prosek kumulativne sirove nagrade G po epizodi."""
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
# DEO 4 — REINFORCE (gradijent politike, softmaks parametrizacija)
# =============================================================================

def _softmax(logits):
    """Numerički stabilan softmaks: oduzima max da spreči overflow."""
    m    = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s    = sum(exps)
    return [e / s for e in exps]


def reinforce(sim, n_ep, gamma=GAMMA, alpha_fn=None, snapshot_every=200):
    """
    REINFORCE sa softmaks politikom.
    Parametrizacija: θ[s,a] za svako neterminalno s i akciju a (7×4 = 28 param.).
    π_θ(a|s) = softmax_a(θ[s,·])
    Ažuriranje: θ[s,a'] += α · v_τ · (𝟙[a'=a] − π(a'|s))  ∀a'
    NE sme da koristi sim.model() — samo reset() i step().
    """
    if alpha_fn is None:
        def alpha_fn(e):
            return math.log(e + 1) / (e + 1)

    theta      = {(s, a): 0.0 for s in NON_TERMINAL for a in ACTIONS}
    ep_rewards = []
    theta_hist = []

    for e in range(1, n_ep + 1):
        # --- 1) Generiši epizodu prateći trenutnu politiku ---
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

        # --- 2) Return-to-go unazad: v_τ = r_τ + γ·v_{τ+1} ---
        returns = [0.0] * len(traj)
        v       = 0.0
        for t in reversed(range(len(traj))):
            v          = traj[t][2] + gamma * v
            returns[t] = v

        # --- 3) Ažuriranje parametara ---
        alpha = alpha_fn(e)
        for t, (st, at, _) in enumerate(traj):
            probs = _softmax([theta[(st, a)] for a in ACTIONS])
            for j, aj in enumerate(ACTIONS):
                # skor funkcija: ∂/∂θ[st,aj] ln π = 𝟙[aj=at] − π(aj|st)
                grad = (1.0 if aj == at else 0.0) - probs[j]
                theta[(st, aj)] += alpha * returns[t] * grad

        ep_rewards.append(sum(r for _, _, r in traj))
        if e % snapshot_every == 0:
            theta_hist.append(dict(theta))

    return theta, ep_rewards, theta_hist


def _policy_from_theta(theta):
    """Greedy politika iz θ: argmax_a π_θ(a|s) za svako neterminalno s."""
    pi = {}
    for s in NON_TERMINAL:
        probs = _softmax([theta[(s, a)] for a in ACTIONS])
        pi[s] = ACTIONS[probs.index(max(probs))]
    return pi


# =============================================================================
# GRAFICI (G1–G11)
# =============================================================================

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')

# Globalni stil — serif font, bez gornjih/desnih osa, tačkasti grid
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

# Paleta za linije (8 boja, perceptualno razdvojene, tamnije od tab10)
_PAL8 = ['#1b4f72', '#c0392b', '#1e8449', '#884ea0',
          '#d68910', '#148f77', '#7f8c8d', '#2e86c1']

# Boje po akciji za G8 — teal/crvena/zelena/ljubičasta
_ACT_CLR = {'gore': '#1a6b8a', 'dole': '#c0392b',
             'levo': '#27ae60', 'desno': '#8e44ad'}


def _ensure_out():
    os.makedirs(_OUT, exist_ok=True)


def _draw_grid(ax, V, pi, title, v_min=-1.0, v_max=1.0):
    """2×5 mreza: boja = V vrednost (RdBu_r), simbol = politika."""
    arrows = {'gore': '↑', 'dole': '↓', 'levo': '←', 'desno': '→'}
    term_v = {'B1': -1.0, 'B3': +1.0, 'B5': -1.0}
    cmap   = plt.cm.RdBu_r   # crvena=-1, bela=0, plava=+1

    for ri, row in enumerate(['A', 'B']):
        for ci in range(5):
            s   = f'{row}{ci + 1}'
            y   = 1 - ri
            val = term_v.get(s, V.get(s, 0.0))
            norm = np.clip((val - v_min) / (v_max - v_min), 0.0, 1.0)
            clr  = cmap(norm)
            # tekst crn na svetlim ćelijama, bel na tamnim
            txt_clr = 'white' if (norm < 0.25 or norm > 0.75) else '#1a1a1a'

            ax.add_patch(plt.Rectangle((ci, y), 1, 1,
                                       facecolor=clr, edgecolor='#555', lw=1.2))
            # ime stanja — mali tag u gornjem levom uglu
            ax.text(ci + 0.08, y + 0.88, s,
                    ha='left', va='top', fontsize=7, color=txt_clr, alpha=0.7)
            # vrednost u sredini
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
    ax.set_yticklabels(['red B', 'red A'], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_facecolor('white')   # grid ćelije na beloj pozadini


def _rolling_mean(arr, window):
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def _ql_error_curve(V_star, n_seeds, n_ep, gamma, eps, alpha_fn, snap_every):
    """Vrati niz [n_seeds x n_snaps] max-norm gresaka ||V_t - V*||∞."""
    sim     = Simulator()
    n_snaps = n_ep // snap_every
    errs    = np.zeros((n_seeds, n_snaps))
    for seed in range(n_seeds):
        random.seed(seed)
        _, snaps, _ = q_learning(sim, n_ep, gamma, eps, alpha_fn, snap_every)
        for t, snap in enumerate(snaps):
            errs[seed, t] = max(abs(snap[s] - V_star[s]) for s in NON_TERMINAL)
    return errs


# ── G1 ────────────────────────────────────────────────────────────────────────

def make_g1(V_star, pi_star):
    """G1: V* heatmap + strelice politike na 2x5 mrezi."""
    _ensure_out()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    _draw_grid(ax, V_star, pi_star, 'Etalon: optimalna vrednosna funkcija V*(s) i politika π*(s)  [γ=0.9]')
    plt.tight_layout()
    path = os.path.join(_OUT, 'G1_Vstar_pistar.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Sacuvano: {path}')


# ── G2 ────────────────────────────────────────────────────────────────────────

def make_g2(V_hist, V_star):
    """G2: konvergencija iteracije Q-vrednosti — V_t(s) po iteraciji."""
    _ensure_out()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, s in enumerate(NON_TERMINAL):
        col = _PAL8[i % len(_PAL8)]
        ys  = [snap[s] for snap in V_hist]
        ax.plot(ys, color=col, label=s, linewidth=2.0)
        ax.axhline(V_star[s], color=col, linestyle='--', linewidth=1.0, alpha=0.45)
    ax.set_xlabel('Iteracija DP')
    ax.set_ylabel('$V_t(s)$')
    ax.set_title('Konvergencija iteracije Q-vrednosti po stanjima  (γ = 0.9)\n'
                 'Pune linije: $V_t(s)$        Isprekidane: $V^*(s)$')
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    path = os.path.join(_OUT, 'G2_QVI_konvergencija.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Sacuvano: {path}')


# ── G3 ────────────────────────────────────────────────────────────────────────

def make_g3(V_snaps, V_star, snap_every, gamma=0.9, eps=0.1):
    """G3: V_t(s) -> V*(s) za svih 7 stanja — glavni trazeni grafik."""
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
        ax.set_title(f'Stanje  {s}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Epizoda', fontsize=8)
        ax.set_ylabel('V', fontsize=8)
        ax.legend(fontsize=7.5)
    axes[7].axis('off')
    axes[7].text(0.5, 0.55,
                 f'Q-učenje\nγ = {gamma},  ε = {eps}\n'
                 f'Isprekidano = $V^*$ (DP etalon)',
                 ha='center', va='center', fontsize=10,
                 transform=axes[7].transAxes,
                 bbox=dict(boxstyle='round,pad=0.6',
                           facecolor='#eaf4fb', edgecolor='#2e86c1',
                           linewidth=1.2, alpha=0.9))
    fig.suptitle('Q-učenje: konvergencija $V_t(s)$ ka etalonu $V^*(s)$ po stanjima',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(_OUT, 'G3_QL_Vt_po_stanjima.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Sacuvano: {path}')


# ── G4 ────────────────────────────────────────────────────────────────────────

def make_g4(V_star, n_seeds=20, n_ep=10_000, snap_every=100, gamma=0.9, eps=0.1):
    """G4: poredjenje stopa ucenja alpha — ||V_t - V*|| po epizodi."""
    _ensure_out()

    def af_var(e):   return math.log(e + 1) / (e + 1)
    def af_005(_e):  return 0.05
    def af_010(_e):  return 0.10
    def af_020(_e):  return 0.20
    def af_050(_e):  return 0.50

    linestyles = ['-', '--', '-.', ':', '-']
    linewidths = [2.4, 1.9, 1.9, 1.9, 1.5]
    schemes = [('log(e+1)/(e+1)', af_var), ('α=0.05', af_005),
               ('α=0.10', af_010), ('α=0.20', af_020), ('α=0.50', af_050)]
    x = np.arange(1, n_ep // snap_every + 1) * snap_every

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for idx, ((name, alpha_fn), ls, lw) in enumerate(
            zip(schemes, linestyles, linewidths)):
        col = _PAL8[idx % len(_PAL8)]
        print(f'    G4: {name} ({n_seeds} seedova)...')
        errs = _ql_error_curve(V_star, n_seeds, n_ep, gamma, eps, alpha_fn, snap_every)
        m, s = errs.mean(0), errs.std(0)
        ax.plot(x, m, color=col, label=name, linewidth=lw, linestyle=ls)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.12)

    ax.set_yscale('log')   # log skala — bolje prikazuje razlike platoa
    ax.set_xlabel('Epizoda')
    ax.set_ylabel('$\\|V_t - V^*\\|_\\infty$  (log skala)')
    ax.set_title(f'Brzina konvergencije Q-učenja za različite stope učenja α\n'
                 f'(γ={gamma}, ε={eps}, {n_seeds} izvođenja, senka = ±std)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(_OUT, 'G4_QL_alpha_poredjenje.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Sacuvano: {path}')


# ── G5 ────────────────────────────────────────────────────────────────────────

def make_g5(V_star, n_seeds=20, n_ep=10_000, snap_every=100, gamma=0.9):
    """G5: poredjenje vrednosti epsilon — ||V_t - V*|| po epizodi."""
    _ensure_out()

    def af_var(e):  return math.log(e + 1) / (e + 1)

    epsilons   = [0.0, 0.05, 0.1, 0.2, 0.3]
    linestyles = [':', '-.', '--', '-', '-']
    linewidths = [1.8, 1.8, 2.0, 2.0, 2.0]
    x          = np.arange(1, n_ep // snap_every + 1) * snap_every

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for idx, (eps, ls, lw) in enumerate(zip(epsilons, linestyles, linewidths)):
        col = _PAL8[idx % len(_PAL8)]
        print(f'    G5: ε={eps} ({n_seeds} seedova)...')
        errs = _ql_error_curve(V_star, n_seeds, n_ep, gamma, eps, af_var, snap_every)
        m, s = errs.mean(0), errs.std(0)
        ax.plot(x, m, color=col, label=f'ε = {eps}', linewidth=lw, linestyle=ls)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.12)

    ax.set_yscale('log')
    ax.set_xlabel('Epizoda')
    ax.set_ylabel('$\\|V_t - V^*\\|_\\infty$  (log skala)')
    ax.set_title(f'Uticaj parametra istraživanja ε na konvergenciju Q-učenja\n'
                 f'(γ={gamma}, α=log(e+1)/(e+1), {n_seeds} izvođenja)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(_OUT, 'G5_QL_epsilon_poredjenje.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Sacuvano: {path}')


# ── G6 ────────────────────────────────────────────────────────────────────────

def make_g6(V09, pi09, avg09, V999, pi999, avg999):
    """G6: naucene politike za gamma=0.9 i gamma=0.999."""
    _ensure_out()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    _draw_grid(axes[0], V09,  pi09,
               f'Q-ucenje γ=0.9  (prosek/1000 ep = {avg09:+.3f})')
    _draw_grid(axes[1], V999, pi999,
               f'Q-ucenje γ=0.999  (prosek/1000 ep = {avg999:+.3f})')
    fig.suptitle('Naučene politike Q-učenja za različite faktore diskontovanja',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(_OUT, 'G6_QL_gamma_poredjenje.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Sacuvano: {path}')


# ── G7 ────────────────────────────────────────────────────────────────────────

def make_g7(ep_rewards, window=300):
    """G7: REINFORCE — sirova nagrada po epizodi + klizni prosek."""
    _ensure_out()
    x  = np.arange(1, len(ep_rewards) + 1)
    rm = _rolling_mean(np.array(ep_rewards, dtype=float), window)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, ep_rewards, alpha=0.08, color='#7fb3d3',
               s=2, label='Nagrada (sirova)', rasterized=True)
    ax.plot(np.arange(window, len(ep_rewards) + 1), rm,
            color='#c0392b', linewidth=2.2,
            label=f'Klizni prosek (prozor = {window})')
    ax.axhline(0, color='#555', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Epizoda')
    ax.set_ylabel('Ukupna (sirova) nagrada')
    ax.set_title('REINFORCE: ukupna nagrada po epizodi i klizni prosek  (γ = 0.9)',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(_OUT, 'G7_RF_kriva_ucenja.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Sacuvano: {path}')


# ── G8 ────────────────────────────────────────────────────────────────────────

def make_g8(theta_hist, snap_every):
    """G8: REINFORCE — theta parametri i verovatnoce pi po stanjima."""
    _ensure_out()
    x    = np.arange(1, len(theta_hist) + 1) * snap_every
    clrs = _ACT_CLR   # teal/crvena/zelena/ljubičasta

    for fname, ylabel, val_fn, extra_fn, suptitle in [
        ('G8a_RF_theta.png', 'θ[s,a]',
         lambda snap, s, a: snap[(s, a)],
         lambda ax: None,
         'G8a — REINFORCE: parametri θ[s,a] tokom ucenja'),
        ('G8b_RF_pi.png', 'π(a|s)',
         lambda snap, s, a: _softmax([snap[(s, aa)] for aa in ACTIONS])[ACTIONS.index(a)],
         lambda ax: ax.axhline(0.25, color='gray', linestyle=':', linewidth=0.9),
         'G8b — REINFORCE: verovatnoce π(a|s) tokom ucenja'),
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
            ax.set_title(f'Stanje {s}', fontsize=10)
            ax.set_xlabel('Epizoda', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=0.3)
        axes[7].axis('off')
        fig.suptitle(suptitle, fontsize=11, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(_OUT, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Sacuvano: {path}')


# ── G9 ────────────────────────────────────────────────────────────────────────

def make_g9(n_seeds=10, n_ep=20_000, gamma=0.9, window=300):
    """G9: REINFORCE — poredjenje stopa ucenja (klizni prosek nagrade)."""
    _ensure_out()

    def af_var(e):  return math.log(e + 1) / (e + 1)
    def af_001(_e): return 0.001
    def af_01(_e):  return 0.01
    def af_05(_e):  return 0.05

    linestyles = ['-', '--', '-.', ':']
    linewidths = [2.3, 1.9, 1.9, 1.9]
    schemes    = [('log(e+1)/(e+1)', af_var), ('α=0.001', af_001),
                  ('α=0.01', af_01), ('α=0.05', af_05)]
    sim = Simulator()

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for idx, ((name, alpha_fn), ls, lw) in enumerate(
            zip(schemes, linestyles, linewidths)):
        col = _PAL8[idx % len(_PAL8)]
        print(f'    G9: {name} ({n_seeds} seedova)...')
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
    ax.set_xlabel('Epizoda')
    ax.set_ylabel(f'Klizni prosek nagrade  (prozor = {window})')
    ax.set_title(f'REINFORCE: uticaj stope učenja α na brzinu konvergencije\n'
                 f'(γ = {gamma}, {n_seeds} izvođenja, senka = ±std)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(_OUT, 'G9_RF_alpha_poredjenje.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Sacuvano: {path}')


# ── G10 ───────────────────────────────────────────────────────────────────────

def make_g10(theta):
    """G10: REINFORCE — naučena politika na mreži 2×5."""
    _ensure_out()
    pi_rf = _policy_from_theta(theta)
    V_rf  = {s: max(_softmax([theta[(s, a)] for a in ACTIONS]))
             for s in NON_TERMINAL}
    fig, ax = plt.subplots(figsize=(9, 3.5))
    _draw_grid(ax, V_rf, pi_rf,
               'REINFORCE: naučena politika  (vrednost ćelije = max π(a|s))',
               v_min=0.25, v_max=1.0)
    plt.tight_layout()
    path = os.path.join(_OUT, 'G10_RF_politika.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Sacuvano: {path}')


# ── G11 ───────────────────────────────────────────────────────────────────────

def make_g11(V_star, pi_star, Q_ql, theta):
    """G11: sumarno poredjenje DP / Q-ucenje / REINFORCE."""
    _ensure_out()
    pi_ql = {s: _argmax_a(Q_ql, s) for s in NON_TERMINAL}
    V_ql  = {s: max(Q_ql.get((s, a), 0.0) for a in ACTIONS) for s in NON_TERMINAL}
    pi_rf = _policy_from_theta(theta)
    V_rf  = {s: max(_softmax([theta[(s, a)] for a in ACTIONS]))
             for s in NON_TERMINAL}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    _draw_grid(axes[0], V_star, pi_star,
               'DP — Iteracija Q-vrednosti\n(V*, π* — etalon)')
    _draw_grid(axes[1], V_ql,   pi_ql,
               'Q-ucenje\n(γ=0.9, α=log/(e+1), 10 000 ep.)')
    _draw_grid(axes[2], V_rf,   pi_rf,
               'REINFORCE\n(γ=0.9, α=log/(e+1), 30 000 ep.)',
               v_min=0.25, v_max=1.0)
    fig.suptitle('Sumarno poređenje naučenih politika: DP etalon / Q-učenje / REINFORCE',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(_OUT, 'G11_sumarno.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Sacuvano: {path}')


# =============================================================================
# Pokretanje eksperimenata i generisanje grafika  (python homework_3.py)
# =============================================================================

if __name__ == '__main__':
    # ── Deo 1: verifikacija modela prelaza ───────────────────────────────────
    dist = transition_dist('A1', 'desno')
    ok   = all(abs(dist.get(k, 0) - v) < 1e-9
               for k, v in {'A2': 0.6, 'A1': 0.3, 'B1': 0.1}.items())
    print(f"[DEO 1] verifikacija prelaza A1/desno: {ok}  {dist}")

    # ── Deo 2: iteracija Q-vrednosti ─────────────────────────────────────────
    print("\n[DEO 2] Iteracija Q-vrednosti ...")
    Q, V_star, pi_star, V_hist = q_value_iteration(gamma=0.9)
    print(f"  γ=0.9  → konvergencija za {len(V_hist)} iteracija")
    print(f"  V*: { {s: round(V_star[s], 3) for s in NON_TERMINAL} }")

    _, V_star_999, pi_star_999, _ = q_value_iteration(gamma=0.999)

    # ── Deo 3: Q-učenje ──────────────────────────────────────────────────────
    random.seed(0)
    sim = Simulator()

    def alpha_var(e):
        return math.log(e + 1) / (e + 1)

    def alpha_const(_e):
        return 0.1

    print("\n[DEO 3] Q-učenje (γ=0.9, ε=0.1, α=log/(e+1), 10 000 ep.) ...")
    Q_ql, V_snaps_var, _ = q_learning(
        sim, n_ep=10_000, gamma=0.9, eps=0.1, alpha_fn=alpha_var, snapshot_every=100
    )
    pi_ql = {s: _argmax_a(Q_ql, s) for s in NON_TERMINAL}
    V_ql  = {s: max(Q_ql[(s, a)] for a in ACTIONS) for s in NON_TERMINAL}
    avg10 = test_policy(sim, Q_ql, n_test=10)
    avg1k = test_policy(sim, Q_ql, n_test=1000)
    print(f"  Greška |V_t-V*|∞ (posl. snapshot): "
          f"{ {s: round(abs(V_snaps_var[-1][s]-V_star[s]),3) for s in NON_TERMINAL} }")
    print(f"  Test greedy (γ=0.9): 10 ep={avg10:+.3f},  1000 ep={avg1k:+.3f}")

    print("\n[DEO 3] Q-učenje (γ=0.9, α=0.1 konst.) ...")
    Q_const, V_snaps_const, _ = q_learning(
        sim, n_ep=10_000, gamma=0.9, eps=0.1, alpha_fn=alpha_const, snapshot_every=100
    )

    print("\n[DEO 3] Q-učenje (γ=0.999, α=log/(e+1)) ...")
    Q_ql999, _, _ = q_learning(
        sim, n_ep=10_000, gamma=0.999, eps=0.1, alpha_fn=alpha_var, snapshot_every=100
    )
    pi_ql999  = {s: _argmax_a(Q_ql999, s) for s in NON_TERMINAL}
    V_ql999   = {s: max(Q_ql999[(s, a)] for a in ACTIONS) for s in NON_TERMINAL}
    avg10_999 = test_policy(sim, Q_ql999, n_test=10)
    avg1k_999 = test_policy(sim, Q_ql999, n_test=1000)
    print(f"  Test greedy (γ=0.999): 10 ep={avg10_999:+.3f},  1000 ep={avg1k_999:+.3f}")

    # ── Deo 4: REINFORCE ─────────────────────────────────────────────────────
    print("\n[DEO 4] REINFORCE (γ=0.9, α=log/(e+1), 30 000 ep.) ...")
    theta, ep_rewards_rf, theta_hist = reinforce(
        sim, n_ep=30_000, gamma=0.9, alpha_fn=alpha_var, snapshot_every=200
    )
    pi_rf    = _policy_from_theta(theta)
    match_ok = sum(pi_rf[s] == pi_star[s] for s in NON_TERMINAL)
    avg_first = sum(ep_rewards_rf[:1000])  / 1000
    avg_last  = sum(ep_rewards_rf[-1000:]) / 1000
    print(f"  Poklapanje s π*: {match_ok}/7")
    print(f"  Prosečna nagrada: prvih 1000 ep={avg_first:+.3f}, posl. 1000 ep={avg_last:+.3f}")

    # =========================================================================
    # Generisanje grafika G1–G11
    # =========================================================================
    print('\n=== Generisanje grafika ===')

    make_g1(V_star, pi_star)
    make_g2(V_hist, V_star)
    make_g3(V_snaps_var, V_star, snap_every=100)

    print('  G4: poredjenje alpha (multi-seed, moze trajati ~1 min)...')
    make_g4(V_star)

    print('  G5: poredjenje epsilon (multi-seed, moze trajati ~1 min)...')
    make_g5(V_star)

    make_g6(V_ql, pi_ql, avg1k, V_ql999, pi_ql999, avg1k_999)
    make_g7(ep_rewards_rf)
    make_g8(theta_hist, snap_every=200)

    print('  G9: REINFORCE alpha (multi-seed, moze trajati ~2 min)...')
    make_g9()

    make_g10(theta)
    make_g11(V_star, pi_star, Q_ql, theta)

    print(f'\nSvi grafici sacuvani u: {_OUT}')
