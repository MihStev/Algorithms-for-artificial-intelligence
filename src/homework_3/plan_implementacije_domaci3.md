# Plan implementacije — 3. domaći (Učenje podsticanjem, 13E053VI)

> Cilj ovog dokumenta: konkretan, izvodljiv plan implementacije i, što je
> jednako važno, **plan rezultata** — koji grafici se prave, **zašto baš ti**
> i kako se čitaju. Svaka odluka je vezana za teoriju iz konteksta. Kôd nije
> pisan u celini; tamo gde logika nije očigledna dat je pseudo-Python.

---

## 0. Velika slika — kako se 4 celine spajaju

```
                    ┌─────────────────────────────┐
                    │   ZAJEDNIČKI SLOJ (env)      │
                    │  stanja, akcije, move(),     │
                    │  transition_dist(s,a),       │
                    │  reward(s'), Simulator       │
                    └──────────────┬──────────────┘
            koristi MODEL P(s'|s,a)│       koristi SAMO step()/reset()
        ┌──────────────────────────┼───────────────────────────┐
        ▼                          ▼                            ▼
 (2) Iteracija            (3) Q-učenje                  (4) REINFORCE
   Q-vrednosti            (model-free, TD)              (model-free, PG)
   = ETALON (V*, π*)      uči Q → greedy π              uči π_θ direktno
        │                          │                            │
        └──────────► poredi V_t i π sa etalonom ◄───────────────┘
```

Ključ celog domaćeg: **deo (2) daje „tačan odgovor" (V\*, π\*)** jer koristi
poznat model i radi egzaktnu dinamičku optimizaciju. Delovi (3) i (4) **ne smeju**
da koriste model — oni uče iz iskustva, a **kvalitet učenja merimo poređenjem sa
etalonom iz (2)**. To je okosnica svih grafika.

---

## 1. Zajednički sloj: okruženje + simulator (deo 1)

### 1.1 Reprezentacija

- Stanja: stringovi `'A1'..'A5'`, `'B1'..'B5'`. Mapiranje u indekse 0–9 za nizove.
- `NON_TERMINAL = ['A1','A2','A3','A4','A5','B2','B4']` (7)
- `TERMINAL = ['B1','B3','B5']` (3)
- `ACTIONS = ['gore','dole','levo','desno']`
- γ = 0.9 podrazumevano (parametrizovati: 0.9, 0.999).

### 1.2 Deterministički pomeraj i model prelaza

Logika koja sve nosi — najpre deterministički „nameravani" pomeraj, pa od njega
stohastički model. **Zid → ostani u istom stanju.**

```python
def move(s, smer):
    # vrati ciljno stanje za jedan deterministički pomeraj; zid -> isto stanje s
    red, kol = s[0], int(s[1])          # 'A2' -> ('A', 2)
    if smer == 'gore':  red2, kol2 = ('A', kol) if red=='B' else ('A', kol)  # A-red -> zid
    if smer == 'gore':  return s if red=='A' else 'A'+str(kol)
    if smer == 'dole':  return s if red=='B' else 'B'+str(kol)
    if smer == 'levo':  return s if kol==1   else red+str(kol-1)
    if smer == 'desno': return s if kol==5   else red+str(kol+1)

def transition_dist(s, a):
    # P(s'|s,a): 0.6 nameravani + 3x0.1 ostali smerovi + 0.1 "ostani"
    P = defaultdict(float)
    for smer in ACTIONS:
        p = 0.6 if smer == a else 0.1
        P[move(s, smer)] += p        # udarac u zid se sam slije u s
    P[s] += 0.1                       # ishod "ostani u mestu"
    return dict(P)                    # zbir == 1.0
```

**Provera (A1, „desno")** mora dati `{'A2':0.6, 'A1':0.3, 'B1':0.1}` — to je
test koji odmah pokazuje da je model ispravan (poklapa se sa primerom iz teksta).

### 1.3 Nagrada i simulator

Nagrada je **prelazna** i zavisi samo od `s'`:

```python
def reward(s_next):
    if s_next == 'B3':           return +1.0
    if s_next in ('B1','B5'):    return -1.0
    return -0.04                  # "living cost" — gura ka kratkim putanjama do +1
```

```python
class Simulator:
    def reset(self):                      # uniformno 1/7 po neterminalnim
        self.s = random.choice(NON_TERMINAL); return self.s
    def step(self, a):
        P = transition_dist(self.s, a)
        s_next = sample_from(P)            # po verovatnoćama
        r = reward(s_next)
        done = s_next in TERMINAL
        self.s = s_next
        return r, s_next, done
    def model(self, s, a):                 # SME da koristi samo deo (2)
        return transition_dist(s, a)
```

> Disciplina koju treba poštovati: `model()` koristi **samo** iteracija
> Q-vrednosti. Q-učenje i REINFORCE smeju da zovu **samo** `reset()` i `step()`.
> To je suština razlike „poznat MPO" vs „nepoznat MPO" iz teorije.

**Zamka (praktična):** rane epizode uz nasumično ponašanje mogu biti jako duge.
Uvesti `MAX_STEPS` (npr. 200) kao osigurač — epizoda se prekine i doprinese
nakupljenom nagradom. Matematički epizoda uvek završava (terminal je dostižan iz
svakog stanja), ali kapa štiti od patoloških dužina.

---

## 2. Deo 2 — Iteracija Q-vrednosti (etalon)

Egzaktna DP nad poznatim modelom. Daje **V\*** i **π\*** koji su referenca za sve
ostalo. Sinhrono ažuriranje (nove vrednosti iz starih).

```python
def q_value_iteration(gamma, tol=1e-12):
    Q = {(s,a): 0.0 for s in NON_TERMINAL for a in ACTIONS}
    V_hist = []                                   # za grafik konvergencije
    while True:
        Qn, delta = {}, 0.0
        for s in NON_TERMINAL:
            for a in ACTIONS:
                q = 0.0
                for s2, p in transition_dist(s, a).items():
                    v2 = 0.0 if s2 in TERMINAL else max(Q[(s2,a2)] for a2 in ACTIONS)
                    q += p * (reward(s2) + gamma * v2)      # r unutar sume (prelazna nagrada)
                Qn[(s,a)] = q
                delta = max(delta, abs(q - Q[(s,a)]))
        Q = Qn
        V_hist.append({s: max(Q[(s,a)] for a in ACTIONS) for s in NON_TERMINAL})
        if delta < tol: break
    V_star  = {s: max(Q[(s,a)] for a in ACTIONS) for s in NON_TERMINAL}
    pi_star = {s: argmax_a(Q, s) for s in NON_TERMINAL}
    return Q, V_star, pi_star, V_hist
```

**Teorijska sidra:**
- `V*(s) = max_a Q*(s,a)`, `π*(s) = argmax_a Q*(s,a)`.
- Konvergencija je **geometrijska sa stopom γ**: `|V_{t+1}-V_t| ≤ γ^t·max|R|`.
  → za **γ=0.999 konvergira mnogo sporije** nego za 0.9 (broj iteracija raste).
  Ovo direktno objašnjava i sporiju konvergenciju Q-učenja pri γ=0.999.

---

## 3. Deo 3 — Q-učenje (TD, off-policy, ε-gramzivo)

```python
def epsilon_greedy(Q, s, eps):
    if random.random() < eps:  return random.choice(ACTIONS)
    return argmax_a(Q, s)              # VAŽNO: nasumično razbijanje izjednačenja (na startu sve = 0)

def q_learning(sim, n_ep, gamma, eps, alpha_fn, snapshot_every=1):
    Q = defaultdict(float)
    V_snaps, ep_rewards = [], []
    for e in range(1, n_ep+1):
        s, done, G = sim.reset(), False, 0.0
        alpha = alpha_fn(e)            # konstantna u okviru epizode, menja se po epizodi
        steps = 0
        while not done and steps < MAX_STEPS:
            a = epsilon_greedy(Q, s, eps)
            r, s2, done = sim.step(a)
            target = r if done else r + gamma * max(Q[(s2,a2)] for a2 in ACTIONS)
            Q[(s,a)] += alpha * (target - Q[(s,a)])
            s, G, steps = s2, G + r, steps + 1
        ep_rewards.append(G)
        if e % snapshot_every == 0:
            V_snaps.append({s: max(Q[(s,a)] for a in ACTIONS) for s in NON_TERMINAL})
    return Q, V_snaps, ep_rewards
```

**Stope učenja koje upoređujemo (zahtev domaćeg):**
- promenljiva: `alpha_fn = lambda e: log(e+1)/(e+1)`
- konstantna: `alpha_fn = lambda e: c` za nekoliko `c` (npr. 0.05, 0.1, 0.2, 0.5).

**Teorijska sidra:**
- Off-policy: cilj koristi `max_{a'}` bez obzira na izabranu akciju → uči **Q\***
  nezavisno od ε (ε utiče samo na *koje* (s,a) posećujemo).
- Konvergencija ka Q\* traži (Robbins–Monro): `Σα=∞` i `Σα²<∞`.
  - `log(e+1)/(e+1)`: `Σ=∞` (dobro) i `Σ²<∞` (dobro) → **konvergira tačno**, ali kasne
    epizode sporo pomeraju Q.
  - konstantno α: `Σ=∞` ali `Σ²=∞` → **ne konvergira u tačku**, Q „treperi" oko Q\*
    sa rezidualnom varijansom ∝ α. Brzo na početku, plato sa šumom kasnije.
  - **Ovo je tačno ono što grafik konvergencije treba da pokaže.**
- ε-gramzivo = balans istraživanje/eksploatacija. **Bitna olakšica ovog okruženja:**
  pošto `reset()` bira početno stanje uniformno po svih 7 neterminalnih, pokrivenost
  stanja je zagarantovana i bez velikog ε; ε je tu prevashodno da bismo isprobali
  *alternativne akcije* u svakom stanju. Zato očekujemo da je umeren ε (npr. 0.1)
  dovoljan — i to ćemo i pokazati eksperimentom.

**Test naučene politike (zahtev):** posle učenja, pusti **greedy** politiku
(ε=0, bez ažuriranja) kroz **10 epizoda**, uzmi prosečnu **sirovu** ukupnu nagradu.
Ponovi za **γ=0.999**. (Pošto je 10 epizoda mali uzorak sa velikom varijansom,
dodatno prijaviti i prosek na npr. 1000 epizoda radi stabilne procene — ali
„zvanični" broj ostaje 10 kako traži tekst.)

---

## 4. Deo 4 — REINFORCE (gradijent politike)

### 4.1 Parametrizacija (obrazloženje za izveštaj)

Akcije su **diskretne** → prirodan izbor je **softmaks po (stanje, akcija)**:
po jedan logit `θ[s,a]` za svako neterminalno stanje (7×4 = **28 parametara**).

```
π_θ(a|s) = softmax_a( θ[s,·] ) = exp(θ[s,a]) / Σ_{a'} exp(θ[s,a'])
∂/∂θ[s,a'] ln π_θ(a|s) = 𝟙[a'=a] − π_θ(a'|s)      # skor; za druga stanja = 0
```

Zašto softmaks (a ne Gausova kao u vežbama): Gaus je za kontinualne akcije; ovde su
akcije diskretne. Softmaks daje **po parametar za svaki par (s,a) u neterminalnim
stanjima**, što se lepo iscrtava (a baš to domaći i traži). Init `θ=0` → uniformna
politika (1/4 svaka akcija) = dobar, neutralan start.

### 4.2 Algoritam (return-to-go)

```python
def reinforce(sim, n_ep, gamma, alpha_fn):
    theta = {(s,a): 0.0 for s in NON_TERMINAL for a in ACTIONS}
    ep_rewards, theta_hist = [], []
    for e in range(1, n_ep+1):
        # 1) generiši epizodu prateći trenutnu politiku
        traj, s, done, steps = [], sim.reset(), False, 0
        while not done and steps < MAX_STEPS:
            probs = softmax([theta[(s,a)] for a in ACTIONS])
            a = sample(ACTIONS, probs)
            r, s2, done = sim.step(a)
            traj.append((s, a, r)); s, steps = s2, steps+1
        # 2) povraćaji-do-kraja unazad: v_τ = r_τ + γ v_{τ+1}
        v, returns = 0.0, [0.0]*len(traj)
        for τ in reversed(range(len(traj))):
            v = traj[τ][2] + gamma*v; returns[τ] = v
        # 3) ažuriranje: θ += α v_τ ∇ ln π
        alpha = alpha_fn(e)
        for τ, (s,a,_) in enumerate(traj):
            probs = softmax([theta[(s,aa)] for aa in ACTIONS])
            for j, aj in enumerate(ACTIONS):
                grad = (1.0 if aj==a else 0.0) - probs[j]
                theta[(s,aj)] += alpha * returns[τ] * grad
        ep_rewards.append(sum(r for _,_,r in traj))     # SIROVA suma (zahtev)
        theta_hist.append(dict(theta))
    return theta, ep_rewards, theta_hist
```

**Teorijska sidra:**
- Cilj: maksimizirati `E[povraćaj]`; ažuriranje ide uz **skor funkciju** ∇lnπ,
  ponderisano povraćajem `v_τ`.
- Monte-Carlo procena gradijenta je **nepristrasna ali visoke varijanse** (nema
  bootstrappinga ni bazne linije) → kriva nagrade po epizodama je **šumovita**.
- `return-to-go` (umesto pune sume G po svakom koraku) već smanjuje varijansu jer
  izbacuje prošle nagrade (čije je očekivano dejstvo na gradijent 0).
- (Opciono, kao smer ka akter-kritičaru) oduzimanje **bazne linije** b (npr. prosek
  povraćaja) ili standardizacija povraćaja dodatno smanjuje varijansu — može se
  pomenuti/isprobati, nije obavezno.
- Stope učenja (zahtev „kao kod Q-učenja"): isto `log(e+1)/(e+1)` vs konstanta.
  Napomena: pošto `v_τ` može biti većeg modula, REINFORCE često traži **manje** α
  nego Q-učenje (preveliko α → nestabilnost / prerano „zaglavljivanje" u
  determinističku, možda suboptimalnu politiku).

---

## 5. Eksperimenti — kompletna lista (šta tačno pokrećemo)

| # | Eksperiment | Parametri | Što merimo |
|---|-------------|-----------|------------|
| E1 | Etalon | γ=0.9 | V\*, π\*, kriva konvergencije DP |
| E2 | Q-učenje: izbor ε | ε∈{0.0, 0.05, 0.1, 0.2, 0.3}, α=log/(e+1) | ‖V_t−V\*‖ vs epizoda |
| E3 | Q-učenje: α promenljivo vs konstantno | α=log(e+1)/(e+1) vs c∈{0.05,0.1,0.2,0.5}, ε* iz E2 | ‖V_t−V\*‖ vs epizoda |
| E4 | Q-učenje: praćenje V_t(s) | najbolja podešavanja | V_t(s) po stanju vs V\*(s) |
| E5 | Q-učenje: γ poređenje | γ=0.9 i γ=0.999 | π, prosečna nagrada/10 epizoda |
| E6 | REINFORCE: kriva nagrade | γ=0.9, izabrano α | suma sirovih nagrada vs epizoda |
| E7 | REINFORCE: parametri | γ=0.9 | θ[s,a] i π(a|s) po neterminalnim stanjima |
| E8 | REINFORCE: stope učenja | α∈{log/(e+1), 0.001, 0.01, 0.05} | kriva nagrade (klizni prosek) |
| E9 | Sumarno poređenje | sve | π i vrednosti: DP vs Q-uč. vs REINFORCE |

**Reproduktivnost / smanjenje šuma:** za krive poređenja (E2, E3, E8) pokretati
**više nasumičnih seedova** (npr. 20–30) i crtati **srednju vrednost ± std (senka)**.
Jedna trajektorija je previše šumovita da bi se na njoj donosio zaključak o
„brzini konvergencije". Fiksirati seed za finalne prikaze.

**Broj epizoda (orijentaciono):** Q-učenje 5.000–20.000 (mali prostor, brzo);
REINFORCE 20.000–100.000 (veća varijansa, sporije). Tuningovati.

---

## 6. Rezultati i grafici — KOJI, ZAŠTO i kako se čitaju

Ovo je deo na koji se najviše oslanja izveštaj. Za svaki grafik: šta prikazuje,
**zašto baš taj** (vezano za teoriju i za eksplicitne zahteve teksta), kako ga čitati.

### G1 — Etalon: V\* (heatmap) + π\* (strelice) na mreži 2×5
- **Što:** mreža sa bojom = V\*(s), tekst = vrednost, strelica = π\*(s);
  terminali B1/B3/B5 obeleženi ±1.
- **Zašto:** referentni „tačan odgovor" iz dela (2). Sve ostalo poredimo s njim;
  bez ovog grafika ne možemo tvrditi da je nešto „naučeno tačno".
- **Čitanje:** vrednosti rastu ka B3 (+1); strelice u svakom stanju treba da
  „teku" ka B3 i da beže od B1/B5.

### G2 — Konvergencija iteracije Q-vrednosti: V_t(s) vs t
- **Što:** 7 krivih (po stanju) kako V_t raste do V\*.
- **Zašto:** demonstrira da DP konvergira i **koliko brzo** — uvod u priču o γ:
  ista metoda za γ=0.999 traje znatno više iteracija (geometrijska stopa γ).
  Postavlja teorijski okvir za kasniju γ-diskusiju.
- **Čitanje:** brzo zasićenje za γ=0.9; (ako se doda γ=0.999 krivama) primetno
  sporije.

### G3 — Q-učenje: V_t(s) → V\*(s) (7 podgrafika) — **glavni traženi grafik**
- **Što:** za svako neterminalno stanje jedan podgrafik: puna linija = V_t(s) iz
  Q-učenja, isprekidana horizontala = V\*(s) iz dela (2).
- **Zašto:** **tekst doslovno traži**: „prikažite kako se sa iteracijama t menjaju
  V_t(s)=max_a Q_t(s,a) i uporedite sa iteracijom Q-vrednosti". Pokazuje da TD
  bootstrapping privlači procene ka egzaktnom optimumu.
- **Čitanje:** svaka puna linija prilazi svojoj isprekidanoj → algoritam radi.
  Ako neka stalno odstupa → premalo posećeno stanje / loš ε ili α.

### G4 — Q-učenje: agregatna greška ‖V_t−V\*‖∞ vs epizoda (α promenljivo vs konstantno) — **traženo poređenje**
- **Što:** jedna skalarna kriva greške po epizodi za promenljivo α i za nekoliko
  konstantnih α (sve usrednjeno po seedovima, sa senkom std).
- **Zašto:** **najčistiji prikaz „brzine konvergencije"** koji tekst eksplicitno
  traži. Skalar je merljiv i uporediv, za razliku od 7 odvojenih krivih.
- **Čitanje:** očekivano — konstantno α: brz pad pa **plato sa šumom** (rezidualna
  varijansa ∝ α, jer `Σα²=∞`); promenljivo `log(e+1)/(e+1)`: sporiji ali
  **niži/tačniji** kraj (zadovoljava Robbins–Monro). Tačno preslikava teoriju.

### G5 — Q-učenje: izbor ε (greška vs epizoda za više ε)
- **Što:** ‖V_t−V\*‖ za ε∈{0, 0.05, 0.1, 0.2, 0.3}.
- **Zašto:** tekst traži „ε odrediti kroz eksperimente" → grafik je opravdanje
  izbora. Ujedno ilustruje da je zbog uniformnog početnog stanja okruženje
  „blago" prema istraživanju.
- **Čitanje:** ε=0 može zaostati (ne proba alternativne akcije); preveliki ε troši
  korake; biramo ε gde greška najbrže i najniže pada (verovatno ≈0.1).

### G6 — Q-učenje: naučena politika za γ=0.9 i γ=0.999 (dve mreže) + tabela
- **Što:** dve mreže sa strelicama (γ=0.9, γ=0.999); tabela prosečne ukupne nagrade
  na 10 test epizoda (po želji i na 1000) za oba γ.
- **Zašto:** tekst traži prikaz konačne politike i poređenje γ=0.9 vs 0.999 sa
  tumačenjem. Politika kao mreža je najjasnija.
- **Čitanje / tumačenje (teorija):** γ bliže 1 = **dalekovidiji** agent: skoro
  neumanjeno vrednuje +1 i iz udaljenih stanja, spremniji je da plati više
  „living cost" koraka da bi **izbegao rizik** od −1; γ=0.9 je **kratkovidiji**,
  jače umanjuje udaljeni +1, pa relativno više vrednuje brze (moguće rizičnije)
  putanje. U ovako maloj mreži razlika može biti suptilna (kratke putanje), ali
  je tačno to što treba izmeriti i prokomentarisati; takođe naglasiti da γ=0.999
  zahteva **više epizoda** za istu tačnost (veza sa G2).

### G7 — REINFORCE: suma sirovih nagrada po epizodi (+ klizni prosek) — **traženo**
- **Što:** tačke/linija po epizodi + klizni prosek (npr. prozor 200).
- **Zašto:** tekst traži praćenje napretka kroz ukupnu (neponderisanu) nagradu po
  epizodi. Klizni prosek je nužan jer je sirova kriva veoma šumovita (visoka
  varijansa MC gradijenta — teorija).
- **Čitanje:** trend naviše ka optimalnom povraćaju; širina rasipanja ilustruje
  varijansu REINFORCE-a.

### G8 — REINFORCE: parametri politike po neterminalnim stanjima — **traženo**
- **Što:** 7 podgrafika (po jedan po neterminalnom stanju), u svakom 4 krive.
  Prikazati **θ[s,a]** (doslovno „vrednosti parametara politike") i — radi
  tumačenja — i izvedene verovatnoće **π(a|s)**.
- **Zašto:** tekst doslovno traži „vrednosti parametara politike u neterminalnim
  stanjima". Verovatnoće dodatno pokazuju kako politika postaje (skoro)
  deterministička ka optimalnoj akciji.
- **Čitanje:** u svakom stanju jedna akcija „pobeđuje" (njena θ raste, π→1) i
  treba da se poklopi sa π\* iz G1 → potvrda da REINFORCE konvergira ka istom
  optimumu.

### G9 — REINFORCE: poređenje stopa učenja (klizni prosek nagrade) — **traženo**
- **Što:** krive nagrade (klizni prosek) za nekoliko α.
- **Zašto:** tekst traži eksperimentisanje sa stopama učenja.
- **Čitanje:** preveliko α → nestabilno / prerano zaglavljivanje (možda
  suboptimalno); premalo α → sporo; biramo α sa najboljim kompromisom.

### G10 — REINFORCE: naučena politika (mreža strelica) — **traženo**
- **Što:** mreža sa `argmax_a π(a|s)`.
- **Zašto:** prikaz konačne politike (zahtev) i direktno poređenje sa G1 i G6.

### G11 — Sumarno poređenje (tabela/mreža): DP vs Q-učenje vs REINFORCE
- **Što:** uporedna tabela V\*(s) vs V iz Q-učenja, i poklapanje politika sve tri
  metode; tabela prosečnih test nagrada.
- **Zašto:** zatvara priču — **sve tri metode treba da daju istu optimalnu
  politiku** (DP egzaktno, Q-učenje preko vrednosti, REINFORCE direktno preko
  politike). To je najjača potvrda ispravnosti cele implementacije.

> **Minimalni obavezni set** (ako se štedi prostor): G1, G3, G4, G6, G7, G8, G10.
> Ostali su jaka potpora i preporučeni.

---

## 7. Analiza koju izveštaj treba da iznese (povezivanje sa teorijom)

1. **Validacija preko etalona:** Q-učenje (V_t→V\*) i REINFORCE (π→π\*) konvergiraju
   ka rezultatu egzaktne DP → implementacije su korektne.
2. **α promenljivo vs konstantno:** konstantno = brzo + rezidualni šum (ne
   konvergira u tačku, `Σα²=∞`); `log(e+1)/(e+1)` = sporije + tačnije (Robbins–Monro).
   Zaključak: za **finalnu tačnost** promenljivo; za **brz početni napredak** veće
   konstantno; eventualno hibrid (kreni veće, gasi).
3. **ε:** zbog uniformnog početnog stanja istraživanje je manje kritično; umeren ε
   (≈0.1) dovoljan; ε=0 zaostaje jer ne proba alternativne akcije.
4. **γ=0.9 vs 0.999:** efekat horizonta i odnosa „living cost ↔ terminalna nagrada";
   dalekovidost vs rizik kratkih putanja; γ=0.999 sporija konvergencija (γ^t).
   Uporediti i konkretne brojeve prosečne nagrade na 10 (i 1000) epizoda.
5. **REINFORCE varijansa:** šumovita kriva nagrade = posledica MC gradijenta bez
   bootstrappinga/bazne linije; return-to-go i (opciono) bazna linija je smanjuju —
   prirodan most ka akter-kritičaru.

---

## 8. Praktične zamke (lako se previde)

- **Razbijanje izjednačenja u argmax** nasumično — na startu su svi Q=0; bez toga
  agent uvek bira istu (npr. prvu) akciju i učenje je pristrasno.
- **Terminali:** V=0 u terminalu; kad je `done`, cilj u Q-učenju = `r` (bez
  bootstrapa); u DP `max Q[s',·]=0` za terminalno s'.
- **Prelazna nagrada** ulazi **unutar** sume u DP (zavisi od s'), ne ispred.
- **`MAX_STEPS` kapa** epizode (rane duge epizode), inače REINFORCE/Q-učenje vise.
- **Snapshotovanje** V_t i θ na svakih k epizoda (ne svaki korak) — dovoljno za
  glatke krive, jeftinije po memoriji.
- **Više seedova + senka std** za sve krive poređenja; jedan run = nepouzdan
  zaključak o brzini konvergencije.
- **REINFORCE α manje** nego Q-učenje (moduli povraćaja); razmotriti standardizaciju
  povraćaja po epizodi.

---

## 9. Predložena struktura koda

```
env.py          # stanja, akcije, move, transition_dist, reward, Simulator (deo 1)
qvi.py          # q_value_iteration -> V*, π*  (deo 2, etalon)
qlearning.py    # q_learning, epsilon_greedy, α-šeme (deo 3)
reinforce.py    # softmax politika, reinforce (deo 4)
experiments.py  # E1–E9: pokreće sve, sakuplja istorije (seedovi, usrednjavanje)
plots.py        # G1–G11: mreža+strelice, krive konvergencije, itd.
report.(md|html→pdf)  # izveštaj sa graficima i analizom (predaja)
```

**Redosled rada:** (1) env + provera `transition_dist('A1','desno')` →
(2) DP etalon (V\*, π\*) → (3) Q-učenje + E2/E3/E4 grafici uz etalon →
(4) E5 (γ) → (5) REINFORCE + E6/E7/E8 → (6) E9 sumarno → (7) izveštaj.
Predaja: Python kôd + izveštaj (PDF/HTML) sa svim traženim graficima i
**konačnim politikama**, preko MS Teams. **Ne zaboraviti Turn In.**
