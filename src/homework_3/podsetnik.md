# Podsetnik — Učenje podsticanjem (13E053VI, Domaći 3)

> Ovaj fajl pokriva svu teoriju i ceo kod koji smo napisali,  
> bez grafika. Namenjen za ponavljanje ispred ispita.

---

## Sadržaj

1. [Okruženje i simulator](#1-okruženje-i-simulator)
2. [Iteracija Q-vrednosti (etalon)](#2-iteracija-q-vrednosti-etalon)
3. [Q-učenje](#3-q-učenje)
4. [REINFORCE](#4-reinforce)
5. [Ključne razlike i poređenje](#5-ključne-razlike-i-poređenje)

---

## 1. Okruženje i simulator

### Mreža 2×5

```
        col1   col2   col3   col4   col5
row A    A1     A2     A3     A4     A5
row B   B1(-1)  B2    B3(+1)  B4   B5(-1)
```

- **Terminalna stanja:** B1 (nagrada −1), B3 (+1), B5 (−1)
- **Neterminalna stanja:** A1, A2, A3, A4, A5, B2, B4 — ukupno 7
- **Akcije:** gore, dole, levo, desno
- **Početno stanje:** uniformno slučajno iz 7 neterminalnih (1/7 svako)
- **Gamma:** γ = 0.9 (podrazumevano), γ = 0.999 (alternativno)

### Model prelaza (stohastika)

Kada agent izabere akciju `a`:

```
P(s'|s, a) = 0.6   ako s' = nameravani smer
             0.1   za svaki od preostala 3 smera
             0.1   "ostani u mestu" (bez pomeraja)
```

Udarac u zid → agent ostaje na mestu.

**Verifikacija (A1, desno):**
```
P(A2 | A1, desno) = 0.6    ← nameravani smer
P(A1 | A1, desno) = 0.3    ← gore (zid) + levo (zid) + ostani = 3×0.1
P(B1 | A1, desno) = 0.1    ← dole
```

### Nagrada

```
R(s') = +1.0   ako s' == 'B3'
        -1.0   ako s' in ('B1', 'B5')
        -0.04  inače  ← "living cost", gura ka kratkim putanjama
```

Nagrada je **prelazna**: zavisi od narednog stanja `s'`, ne od trenutnog.  
Ovo je važno za implementaciju DP (nagrada ulazi **unutar** sume).

### Kod okruženja

```python
STATES       = [f'A{i}' for i in range(1, 6)] + [f'B{i}' for i in range(1, 6)]
NON_TERMINAL = ['A1', 'A2', 'A3', 'A4', 'A5', 'B2', 'B4']  # 7 stanja
TERMINAL     = ['B1', 'B3', 'B5']                            # 3 stanja
ACTIONS      = ['gore', 'dole', 'levo', 'desno']
GAMMA        = 0.9
MAX_STEPS    = 200   # zaštita od beskonačnih epizoda u ranoj fazi
```

```python
def move(s, smer):
    """Deterministički pomeraj. Zid → vraća isto stanje s."""
    row, col = s[0], int(s[1])
    if smer == 'gore':   return s if row == 'A' else f'A{col}'
    if smer == 'dole':   return s if row == 'B' else f'B{col}'
    if smer == 'levo':   return s if col == 1   else f'{row}{col - 1}'
    return s if col == 5 else f'{row}{col + 1}'   # desno
```

```python
def transition_dist(s, a):
    """
    Vraća rečnik {s': P(s'|s,a)}.
    Koristeći defaultdict, udarci u zid se automatski sabiraju u P[s].
    """
    P = defaultdict(float)
    for smer in ACTIONS:
        p = 0.6 if smer == a else 0.1   # 0.6 nameravani, 0.1 ostali
        P[move(s, smer)] += p            # ako je zid → move vraća s, sabira se u P[s]
    P[s] += 0.1                          # ishod "ostani u mestu"
    return dict(P)                       # zbir uvek = 1.0
```

```python
def reward(s_next):
    if s_next == 'B3':           return  1.0
    if s_next in ('B1', 'B5'):   return -1.0
    return -0.04
```

```python
class Simulator:
    def reset(self):
        """Uniformno bira jedno od 7 neterminalnih stanja."""
        self.s = random.choice(NON_TERMINAL)
        return self.s

    def step(self, a):
        """
        Primeni akciju a.
        Vraća: (nagrada, novo_stanje, da_li_je_epizoda_gotova).
        """
        P      = transition_dist(self.s, a)
        s_next = random.choices(list(P.keys()), weights=list(P.values()))[0]
        r      = reward(s_next)
        done   = s_next in TERMINAL
        self.s = s_next
        return r, s_next, done

    def model(self, s, a):
        """Izlaže P(·|s,a) — sme koristiti SAMO iteracija Q-vrednosti."""
        return transition_dist(s, a)
```

**Ključna disciplina:** Q-učenje i REINFORCE smeju zvati samo `reset()` i `step()`.  
`model()` koristi jedino iteracija Q-vrednosti (Deo 2).

---

## 2. Iteracija Q-vrednosti (etalon)

### Teorija

**Q-vrednost** = očekivani diskontovani prinos pri primeni `a` u `s`, pa optimalno:

```
Q*(s, a) = Σ_{s'} P(s'|s,a) · [R(s') + γ · max_{a'} Q*(s', a')]
```

**Veza sa V:** `V*(s) = max_a Q*(s, a)`,  `π*(s) = argmax_a Q*(s, a)`

**Algoritam (sinhrona iteracija):**
```
Q_0(s, a) = 0  za sve (s, a)

Za t = 1, 2, ... do konvergencije:
    Q_{t+1}(s, a) = Σ_{s'} P(s'|s,a) · [R(s') + γ · max_{a'} Q_t(s', a')]
```

Ažuriranje je **sinhrono** — nove vrednosti se računaju isključivo iz starih `Q_t`.

**Zašto konvergira?**  
Belmanov operator je kontrakcija sa konstantom γ:
```
‖Q_{t+1} − Q*‖∞ ≤ γ · ‖Qt − Q*‖∞ → 0
```
Za γ = 0.9 → konvergira za ~53 iteracije.  
Za γ = 0.999 → konvergira za ~66 iteracije (sporija konvergencija jer je γ blizu 1).

### Kod

```python
def _argmax_a(Q, s):
    """
    Vraća akciju sa maksimalnom Q vrednošću.
    VAŽNO: nasumično razbijanje izjednačenja!
    Na početku su svi Q=0 — bez ovoga uvek bira prvu akciju i uči pristrasno.
    """
    best = max(Q[(s, a)] for a in ACTIONS)
    return random.choice([a for a in ACTIONS if Q[(s, a)] == best])


def q_value_iteration(gamma=GAMMA, tol=1e-12):
    Q      = {(s, a): 0.0 for s in NON_TERMINAL for a in ACTIONS}
    V_hist = []   # lista snimaka V po iteraciji — za grafik konvergencije

    while True:
        Q_new = {}
        delta = 0.0   # maksimalna promena u ovoj iteraciji

        for s in NON_TERMINAL:
            for a in ACTIONS:
                # Belmanova projekcija: nagrada UNUTAR sume (prelazna nagrada!)
                q = sum(
                    p * (reward(s2) + gamma * (
                        0.0 if s2 in TERMINAL          # terminali nemaju budućnost
                        else max(Q[(s2, a2)] for a2 in ACTIONS)
                    ))
                    for s2, p in transition_dist(s, a).items()
                )
                Q_new[(s, a)] = q
                delta = max(delta, abs(q - Q[(s, a)]))  # prati konvergenciju

        Q = Q_new
        V_hist.append({s: max(Q[(s, a)] for a in ACTIONS) for s in NON_TERMINAL})

        if delta < tol:   # konvergencija postignuta
            break

    V_star  = {s: max(Q[(s, a)] for a in ACTIONS) for s in NON_TERMINAL}
    pi_star = {s: _argmax_a(Q, s) for s in NON_TERMINAL}
    return Q, V_star, pi_star, V_hist
```

**Šta vraća:**
- `Q` — optimalne Q vrednosti (28 vrednosti za 7×4 parova)
- `V_star` — optimalne V vrednosti po stanju
- `pi_star` — optimalna politika (akcija po stanju)
- `V_hist` — lista snimaka za grafik konvergencije

**Rezultati (γ = 0.9):**
```
V*(A1) = +0.265,  V*(A2) = +0.610,  V*(A3) = +0.846
V*(A4) = +0.610,  V*(A5) = +0.265
V*(B2) = V*(B4) = +0.662

π*(A1) = desno,  π*(A2) = desno,  π*(A3) = dole
π*(A4) = levo,   π*(A5) = levo
π*(B2) = desno,  π*(B4) = levo
```

---

## 3. Q-učenje

### Teorija

**Osnovna razlika od DP:** ne koristimo model `P(s'|s,a)`.  
Uči Q vrednosti direktno iz iskustva: osmotrenih četvorki `(s, a, r, s')`.

**Formula ažuriranja (TD update):**
```
Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') − Q(s, a)]
                           └─────────────────────────────────────┘
                                      TD greška δ
```

Ekvivalentno: `Q(s, a) ← (1 − α) · Q(s, a) + α · target`

**Gde je target:**
```
target = r                           ako je s' terminalno (done=True)
target = r + γ · max_{a'} Q(s', a') inače (bootstrapping)
```

**Zašto "off-policy"?**  
Cilj uvek koristi `max_{a'}` bez obzira koja akcija je stvarno izabrana u `s'`.  
Ε-gramziva politika utiče na **pokrivenost** stanja, ali ne na **vrednost** ka kojoj konvergiramo.

### ε-gramzivo istraživanje

```
a = nasumična akcija         sa verovatnoćom ε  (istraživanje)
a = argmax_a Q(s, a)         sa verovatnoćom 1-ε (eksploatacija)
```

**Zašto nasumično razbijanje izjednačenja?**  
Na početku su svi Q=0. Bez nasumičnog izbora, agent uvek bira prvu akciju u listi  
i nikad ne istraži ostale — politika ostaje pristrasna ka prvoj akciji.

### Stope učenja — Robbins-Monro uslovi

Za konvergenciju Q u tačku `Q*` moraju biti zadovoljeni:
```
Σ αₑ = ∞    (dovoljno velikih koraka da se ispravi svaka greška)
Σ αₑ² < ∞   (koraci se smanjuju da bi se eliminisao šum)
```

| Strategija | Σα | Σα² | Konvergencija |
|---|---|---|---|
| `ln(e+1)/(e+1)` | ∞ ✓ | < ∞ ✓ | U tačku ka Q* |
| Konstantno α | ∞ ✓ | = ∞ ✗ | Treperi ± α oko Q* |

**Praktična implikacija:**
- Promenljiva stopa → tačnija konvergencija, sporiji početak
- Konstantna stopa → brž početak, rezidualni šum ∝ α (nikad ne konvergira u tačku)

### Kod

```python
def _epsilon_greedy(Q, s, eps):
    """
    ε-gramziv izbor akcije.
    Nasumično razbijanje izjednačenja sprečava inicijalnu pristrasnost.
    """
    if random.random() < eps:
        return random.choice(ACTIONS)    # istraživanje
    best = max(Q[(s, a)] for a in ACTIONS)
    return random.choice([a for a in ACTIONS if Q[(s, a)] == best])  # eksploatacija


def q_learning(sim, n_ep, gamma=GAMMA, eps=0.1, alpha_fn=None, snapshot_every=50):
    """
    Q-učenje sa ε-gramzivim istraživanjem.

    Parametri:
        sim          -- Simulator (koristi SAMO reset() i step(), nikad model()!)
        n_ep         -- broj epizoda
        gamma        -- faktor diskontovanja
        eps          -- stopa istraživanja
        alpha_fn(e)  -- funkcija koja vraća stopu učenja za epizodu e
        snapshot_every -- svakih koliko epizoda snimamo V vrednosti
    """
    if alpha_fn is None:
        def alpha_fn(e):
            return math.log(e + 1) / (e + 1)

    Q          = defaultdict(float)   # Q(s,a) = 0 za sve parove (defaultdict)
    V_snaps    = []                   # snimci V_t(s) za grafik
    ep_rewards = []                   # ukupna nagrada po epizodi

    for e in range(1, n_ep + 1):
        s     = sim.reset()    # uniformno slučajno neterminalno stanje
        done  = False
        G     = 0.0            # akumulirana nagrada epizode
        steps = 0
        alpha = alpha_fn(e)    # alpha se menja po epizodi, konstantna unutar epizode

        while not done and steps < MAX_STEPS:
            a           = _epsilon_greedy(Q, s, eps)
            r, s2, done = sim.step(a)

            # TD target: bez bootstrapa ako je terminal, sa bootstrapom inače
            target     = r if done else r + gamma * max(Q[(s2, a2)] for a2 in ACTIONS)

            # TD ažuriranje: pomeri Q(s,a) malo ka targetu
            Q[(s, a)] += alpha * (target - Q[(s, a)])

            # Prekoračili smo na s2, nastavimo petlju odatle
            s, G, steps = s2, G + r, steps + 1

        ep_rewards.append(G)
        if e % snapshot_every == 0:
            # Izvuci V_t(s) = max_a Q_t(s,a) za svako neterminalno stanje
            V_snaps.append({s: max(Q[(s, a)] for a in ACTIONS) for s in NON_TERMINAL})

    return dict(Q), V_snaps, ep_rewards
```

**Vizuelno — tok jedne epizode:**
```
s=A2, α=0.05, ε=0.1

korak 1:  s=A2, a=desno (eksploatacija, eps miss)
          sim.step() → r=-0.04, s2=A3, done=False
          target = -0.04 + 0.9 * max Q(A3, ·)  ← bootstrapping!
          Q(A2, desno) += 0.05 * (target - Q(A2, desno))

korak 2:  s=A3, a=dole (eksploatacija)
          sim.step() → r=+1.0, s2=B3, done=True
          target = +1.0                          ← NEMA bootstrapa, terminal!
          Q(A3, dole) += 0.05 * (1.0 - Q(A3, dole))

G = -0.04 + 1.0 = +0.96
```

### Testiranje naučene politike

```python
def test_policy(sim, Q, n_test=10):
    """
    Testira greedy politiku (eps=0) bez ažuriranja Q.
    Vraća prosečnu ukupnu nagradu.
    """
    total = 0.0
    for _ in range(n_test):
        s     = sim.reset()
        done  = False
        G     = 0.0
        steps = 0
        while not done and steps < MAX_STEPS:
            a          = _argmax_a(Q, s)    # greedy, bez istraživanja
            r, s, done = sim.step(a)
            G         += r
            steps     += 1
        total += G
    return total / n_test
```

### Rezultati i zaključci

**Konvergencija V_t(s) → V*(s):**
- Stanja bliža terminalu konvergiraju brže (manje karika u lancu zavisnosti)
- Ugaona stanja (A1, A5) konvergiraju sporije jer se retko posećuju

**Izbor ε:**
- ε=0 zaostaje — agent nikad ne proba alternativne akcije
- ε=0.1 optimalno — umerno istraživanje, brza konvergencija
- ε=0.3 sporije — previše slučajnih akcija troši korake

**Uticaj γ:**
| Parametar | γ=0.9 | γ=0.999 |
|---|---|---|
| DP iteracije do konvergencije | 53 | 66 |
| Prosečna nagrada (1000 ep) | +0.706 | +0.655 (za 10k ep učenja) |
| Dalekovidost agenta | Kratak horizont | Dug horizont |

Dalekovidiji agent (γ bliže 1) više vrednuje buduće nagrade → spremniji da plati  
više „living cost" koraka da bi izbegao terminale −1. Sporija konvergencija jer  
greška se širi geometrijski sa stopom γ^t.

---

## 4. REINFORCE

### Teorija

**Fundamentalna razlika od Q-učenja:**
```
Q-učenje:   iskustvo → Q(s,a) → π(s) = argmax_a Q(s,a)    (indirektan put)
REINFORCE:  iskustvo → θ → π_θ(a|s) direktno               (direktan put)
```

### Parametrizacija politike — Softmaks

Za diskretne akcije, standardna parametrizacija je **softmaks po (stanje, akcija)**:

```
π_θ(a|s) = exp(θ[s,a]) / Σ_{a'} exp(θ[s,a'])
```

- 7 neterminalnih stanja × 4 akcije = **28 parametara** θ[s,a]
- Inicijalizacija θ=0 → uniformna politika π=1/4 (neutralan start)
- Softmax nikad ne daje 0% ni 100% → politika ostaje stohastička

**Zašto NE Gausova politika?**  
Gaus je za kontinualne akcije (npr. ugao volana). Ovde su akcije diskretne kategorije.  
Softmaks je prirodna generalizacija Bernulija na više klasa.

### Teorema o gradijentu politike

Cilj: maksimizovati `J(θ) = E[G_0]` (očekivani prinos).

```
∇_θ J(θ) = E_π [ Σ_τ v_τ · ∇_θ ln π_θ(a_τ | s_τ) ]
```

**Log-trik (score function):**
```
∇_θ π_θ = π_θ · ∇_θ ln π_θ   ← ovo nam daje Monte Carlo procenu gradijenta
```

### Return-to-go — Zašto unazad?

Umesto pune sume G₀ koristimo **return-to-go** v_τ:
```
v_τ = r_τ + γ·r_{τ+1} + γ²·r_{τ+2} + ...  (od koraka τ do kraja)
    = r_τ + γ·v_{τ+1}   (rekurzija, računamo UNAZAD od kraja)
    v_T = 0
```

**Zašto?** Akcija u koraku τ ne može uticati na nagrade PRE τ.  
Korišćenjem pune sume G₀ bi uveli nepotrebnu varijansu od prošlih nagrada.  
Return-to-go: ista nepristrasnost, manja varijansa.

### Skor funkcija za Softmaks — Zatvorena formula

```
∂/∂θ[s, a'] ln π_θ(a | s) = 𝟙[a'=a] − π_θ(a'|s)
```

Interpretacija:
- Za **izabranu akciju** (a' = at): grad = 1 − π(at|s) > 0 → θ raste → verovatnoća raste
- Za **ostale akcije**: grad = 0 − π(a'|s) < 0 → θ pada → verovatnoća opada

### Ažuriranje parametara

```
θ[s_τ, a'] += α · v_τ · (𝟙[a'=at] − π(a'|s_τ))   za sve a'
```

Efekat: dobre epizode (v_τ > 0) pojačavaju izabrane akcije, loše ih oslabljuju.

### Kod

```python
def _softmax(logits):
    """
    Numerički stabilan softmaks.
    Oduzimamo max da sprečimo exp(700) = overflow (inf).
    exp(700−700)=exp(0)=1 je bezbedan. Matematički identično.
    """
    m    = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s    = sum(exps)
    return [e / s for e in exps]


def reinforce(sim, n_ep, gamma=GAMMA, alpha_fn=None, snapshot_every=200):
    """
    REINFORCE sa softmaks politikom.
    NE koristi sim.model() — samo reset() i step().

    Tok algoritma:
      1) Generiši celu epizodu prateći π_θ
      2) Izračunaj return-to-go unazad
      3) Ažuriraj sve θ[s,a] za svaki korak epizode
    """
    if alpha_fn is None:
        def alpha_fn(e):
            return math.log(e + 1) / (e + 1)

    # 28 parametara: po jedan za svaki (neterminalno stanje, akcija) par
    theta      = {(s, a): 0.0 for s in NON_TERMINAL for a in ACTIONS}
    ep_rewards = []
    theta_hist = []

    for e in range(1, n_ep + 1):

        # ── KORAK 1: Generiši celu epizodu ───────────────────────────────────
        traj  = []       # lista trojki (stanje, akcija, nagrada)
        s     = sim.reset()
        done  = False
        steps = 0

        while not done and steps < MAX_STEPS:
            # Izračunaj verovatnoće akcija prema trenutnoj politici
            probs = _softmax([theta[(s, a)] for a in ACTIONS])
            # Sempluj akciju prema tim verovatnoćama (istraživanje je prirodno!)
            a     = random.choices(ACTIONS, weights=probs, k=1)[0]
            r, s2, done = sim.step(a)
            traj.append((s, a, r))
            s, steps = s2, steps + 1

        # Napomena: REINFORCE čeka kraj epizode pre ažuriranja (Monte Carlo).
        # Q-učenje ažurira posle svakog koraka (online TD). To je ključna razlika.

        # ── KORAK 2: Return-to-go unazad ─────────────────────────────────────
        returns = [0.0] * len(traj)
        v       = 0.0
        for t in reversed(range(len(traj))):
            v          = traj[t][2] + gamma * v   # v_t = r_t + γ·v_{t+1}
            returns[t] = v
        # Na kraju epizode v=0 (nema budućnosti).
        # Zadnji korak: v = r_T + γ·0 = r_T
        # Pretposlednji: v = r_{T-1} + γ·r_T
        # ... i tako dalje unazad.

        # ── KORAK 3: Ažuriranje parametara ───────────────────────────────────
        alpha = alpha_fn(e)
        for t, (st, at, _) in enumerate(traj):
            # Ponovo izračunaj verovatnoće za trenutni θ
            # (θ se menja unutar ove petlje, pa ne možemo koristiti probs iz koraka 1)
            probs = _softmax([theta[(st, a)] for a in ACTIONS])

            for j, aj in enumerate(ACTIONS):
                # Skor funkcija: ∂/∂θ[st,aj] ln π = 𝟙[aj=at] - π(aj|st)
                grad = (1.0 if aj == at else 0.0) - probs[j]
                # Ažuriranje: θ += α · v_τ · grad
                theta[(st, aj)] += alpha * returns[t] * grad

        # Sirova ukupna nagrada epizode (neponderisana suma)
        ep_rewards.append(sum(r for _, _, r in traj))

        if e % snapshot_every == 0:
            theta_hist.append(dict(theta))   # snimak za grafik evolucije parametara

    return theta, ep_rewards, theta_hist


def _policy_from_theta(theta):
    """
    Greedy politika iz θ: argmax_a π_θ(a|s) = akcija sa najvišom verovatnoćom.
    """
    pi = {}
    for s in NON_TERMINAL:
        probs = _softmax([theta[(s, a)] for a in ACTIONS])
        pi[s] = ACTIONS[probs.index(max(probs))]
    return pi
```

### Vizuelno — Tok jedne epizode

```
θ = 0 svuda → π(a|s) = 0.25 za sve (a, s)

Epizoda (seed 42):
  s=A2  → sempluj a=desno (0.25)  → r=-0.04, s2=A3
  s=A3  → sempluj a=dole  (0.25)  → r=+1.0,  s2=B3  TERMINAL

Return-to-go UNAZAD:
  korak 1 (A3, dole): v = 1.0 + 0.9·0   = +1.000
  korak 0 (A2, desno): v = -0.04 + 0.9·1.0 = +0.860

Ažuriranje θ:
  Korak 0 → s=A2, a=desno, v=+0.860:
    θ[A2, desno] += α · 0.860 · (1 - 0.25) = +0.645α   ↑ RASTE
    θ[A2, gore]  += α · 0.860 · (0 - 0.25) = -0.215α   ↓ PADA
    θ[A2, dole]  += α · 0.860 · (0 - 0.25) = -0.215α   ↓ PADA
    θ[A2, levo]  += α · 0.860 · (0 - 0.25) = -0.215α   ↓ PADA

  Korak 1 → s=A3, a=dole, v=+1.000 (jak signal!):
    θ[A3, dole]  += α · 1.000 · (1 - 0.25) = +0.750α   ↑ RASTE
    θ[A3, *ostale*] svaka += α · 1.000 · (0 - 0.25) = -0.250α
```

Posle ~30.000 epizoda: θ[A3, dole] postaje jako veliki (→ π(dole|A3) ≈ 91.9%).

### Rezultati

```
Finalna politika (30.000 ep, γ=0.9, α=ln(e+1)/(e+1)):

Stanje   REINFORCE   pi* (DP)   OK?
A1       desno       desno      OK
A2       desno       desno      OK
A3       dole        dole       OK
A4       levo        levo       OK
A5       levo        levo       OK
B2       desno       desno      OK
B4       levo        levo       OK

Verovatnoće finalne politike:
  A3: π(dole) = 91.9%   ← najjači signal, direktan put ka B3
  B2: π(desno) = 81.5%
  B4: π(levo)  = 80.1%
  A1: π(desno) = 81.7%
  ...

Prosečna nagrada: prvih 1000 ep = +0.24, poslednjih 1000 ep = +0.57
```

---

## 5. Ključne razlike i poređenje

### Teorijska poređenja

| Osobina | Iteracija Q-vred. | Q-učenje | REINFORCE |
|---|---|---|---|
| **Model okruženja** | Zahteva P(s'\|s,a) | Nije potreban | Nije potreban |
| **Šta uči** | Q*(s,a) egzaktno | Q*(s,a) aprox. | π_θ(a\|s) direktno |
| **Ažuriranje** | DP, offline, egzaktno | TD, posle svakog koraka | MC, posle cele epizode |
| **Bootstrapping** | Da (V_{t-1}) | Da (max Q(s',·)) | Ne (čeka kraj ep.) |
| **Pristrasnost** | Nema | Ima (bootstrap) | Nema (MC) |
| **Varijansa** | Nema | Niska | Visoka |
| **Brzina** | 53 DP iteracije | ~10.000 ep | ~30.000 ep |
| **Politika** | Deterministička | Deterministička | Stohastička |
| **Poklapanje s π*** | 7/7 (egzaktno) | 7/7 | 7/7 |

### Zašto je REINFORCE šumovitiji od Q-učenja?

```
Q-učenje:   ažurira Q(s,a) POSLE SVAKOG KORAKA → niska varijansa
            ali koristi Q(s',·) kao procenu → pristrasnost (bootstrapping)

REINFORCE:  čeka KRAJ CELE EPIZODE → nema pristrasnosti (čista MC procena)
            ali jedna epizoda ≠ očekivana vrednost → visoka varijansa
```

### Šta smanjuje varijansu u REINFORCE?

1. **Return-to-go** (umesto pune sume G₀) — već implementirano, bez prošlih nagrada
2. **Bazna linija b** — oduzimanje proseka: `(v_τ - b)·∇ln π`, gde b = npr. prosek prinosa
3. **Akter-kritičar** — critic uči V(s), zamenjuje return-to-go: `(r + γV(s') - V(s))·∇ln π`

### Stope učenja — konkretno

**Q-učenje** α=log/(e+1):
```
e=1:      α = ln(2)/2   ≈ 0.347
e=10:     α = ln(11)/11 ≈ 0.218
e=100:    α ≈ 0.046
e=1000:   α ≈ 0.0069
e=10000:  α ≈ 0.00092
```

**REINFORCE** — treba manje α od Q-učenja jer su `|v_τ|` po modulu veći od TD greške `|δ|`.  
Tipično: Q-učenje alpha=0.1 ≈ REINFORCE alpha=0.01

### Zamke koje se lako previde

| Problem | Simptom | Rešenje |
|---|---|---|
| Q=0 na startu, deterministički argmax | Uvek ista akcija | `random.choice` pri izjednačenju |
| Terminal u Q-učenju | Bootstrap gde ga ne treba | `target = r` kad `done=True` |
| Prelazna nagrada u DP | Nagrada izvan sume | `R(s')` unutar `Σ_s' p·(R(s')+γV(s'))` |
| Duge rane epizode | Program visi | `MAX_STEPS = 200` kapa |
| Preveliko α u REINFORCE | Nestabilnost | Koristiti manje α nego za Q-učenje |
| Unicode u print() na Windows | UnicodeEncodeError | `sys.stdout.reconfigure(encoding='utf-8')` |

---

## Brzi podsećaj na formule

```
# Markovljev proces odlučivanja (MPO):
J(π) = E[R(s0) + γ·R(s1) + γ²·R(s2) + ...]

# Belmanova jednačina optimalnosti:
V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s') + γ·V*(s')]
Q*(s,a) = Σ_{s'} P(s'|s,a) [R(s') + γ·max_{a'} Q*(s',a')]

# Iteracija Q-vrednosti (DP):
Q_{t+1}(s,a) = Σ_{s'} P(s'|s,a) [R(s') + γ·max_{a'} Q_t(s',a')]

# Q-učenje (TD, off-policy):
Q(s,a) += α [r + γ·max_{a'} Q(s',a') - Q(s,a)]
          └──────────── TD greška δ ───────────┘

# Softmaks politika (REINFORCE):
π_θ(a|s) = exp(θ[s,a]) / Σ_{a'} exp(θ[s,a'])

# Skor funkcija (softmaks):
∂/∂θ[s,a'] ln π_θ(a|s) = 𝟙[a'=a] - π_θ(a'|s)

# Return-to-go:
v_τ = r_τ + γ·v_{τ+1},  v_T = 0

# REINFORCE ažuriranje:
θ[s,a'] += α · v_τ · (𝟙[a'=at] - π_θ(a'|s_τ))

# Efektivna veličina uzorka (za čestične filtere):
N_eff = 1 / Σ_i (w_i)²
```

---

*Sve što treba za ispit je ovde. Sreću!*
