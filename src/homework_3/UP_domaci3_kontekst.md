# Učenje podsticanjem — kontekst za 3. domaći (13E053VI, ETF Beograd)

> **Svrha ovog fajla.** Ovo je sažetak svih materijala (3 teorijska predavanja,
> 1 vežba, 1 tekst domaćeg) spakovan u jedan fajl. Cilj je da novi chat učita
> kontekst odavde i **ne mora ponovo da otvara/čita PDF-ove**. Sadrži: tačan
> prepis teksta domaćeg, potpunu specifikaciju okruženja, sve potrebne formule
> sa objašnjenjima, pseudokôd za sva 4 algoritma, i rešene zadatke sa vežbi kao
> reference. Notacija: γ = faktor umanjenja, α = stopa učenja, ε = stopa
> istraživanja, θ = parametri politike, π = politika, Σ = suma.
>
> **Napomena o izvornim fajlovima.** Pet „PDF" fajlova zapravo nisu PDF nego
> arhive sa slikama stranica + propratnim tekstom. Veliki teorijski deck
> (`P11b…`, 62 slajda) je skoro u celini slikovni (Berkeley CS188 gridworld
> animacije), pa je teorija ispod rekonstruisana iz slajdova. Treći teorijski
> fajl (`P11c…`, nepotpuno poznati MPO) sadrži samo naslove sekcija + dva rešena
> ispitna zadatka — iz njih su izvučene ključne formule za Q-učenje i REINFORCE.

---

## 1. TEKST DOMAĆEG ZADATKA (prepis)

**ETF Beograd — Katedra za signale i sisteme**
**13E053VI, 3. domaći zadatak 2025/26 — Učenje podsticanjem**

Za jednostavno okruženje opisano u nastavku treba da implementirate:

1. **simulator**,
2. **algoritam iteracije Q-vrednosti**,
3. **algoritam Q-učenja**,
4. **algoritam REINFORCE**.

### Okruženje (Slika 1)

Okruženje je mreža **2×5** (vidi potpunu specifikaciju u sekciji 2). Stanja
**B1, B3 i B5 su terminalna**. Pri prelasku u **B1 i B5** agent dobija nagradu
**−1**, a pri prelasku u **B3** dobija nagradu **+1**, nakon čega se epizoda
završava. U ostalim slučajevima agent dobija nagradu **−0.04** u svakom koraku.
Faktor umanjenja budućih nagrada je **γ = 0.9**, ako se za neku tačku
eksplicitno ne navede drugačija vrednost.

Na početku svake epizode agent kreće iz **slučajno izabranog neterminalnog
stanja**. Raspodela početnih stanja je **uniformna** (verovatnoća za svako je
**1/7**). Okruženje je **potpuno opservabilno** i agent uvek zna u kom stanju se
nalazi.

U svakom stanju agent ima **4 akcije: gore, dole, levo, desno**. Okruženje je
**stohastično**: kada agent odabere akciju, pomera se u izabranom smeru sa
verovatnoćom **0.6**; u protivnom se pomera u nekom od preostala 3 smera ili
ostaje u mestu (kao da nije zadata nikakva akcija), sa **podjednakim
verovatnoćama** (po 0.1). Ukoliko udari u zid, ostaje u istom stanju.

> *Primer iz teksta:* akcija „desno" u stanju A1 → sa verovatnoćom 0.6 prelazi u
> A2; sa verovatnoćom 0.3 udara u zid (gore/levo) ili se ne pomeri, pa u sva tri
> slučaja ostaje u A1; sa verovatnoćom 0.1 prelazi u B1.

### Simulator

Simulatoru se zadaje akcija na osnovu koje on ažurira interno stanje u skladu sa
opisom okruženja i vraća:

- osvojenu **nagradu** u tom koraku,
- **novo stanje**,
- informaciju o tome da li je **epizoda završena**.

Osim toga, za implementaciju **iteracije Q-vrednosti** biće potrebna i
**raspodela po mogućim narednim stanjima** za svaki par (stanje, akcija) — tj.
model prelaza P(s′ | s, a). (Ovo je deo koji simulator interno zna; algoritmi
iteracije Q-vrednosti smeju da ga koriste, dok Q-učenje i REINFORCE ne smeju.)

### Q-učenje (zahtevi)

- Koristite **ε-gramzivo istraživanje**. Adekvatnu vrednost za ε odredite kroz
  eksperimente.
- Teorijski, stopa učenja treba da se smanjuje tokom vremena, ali ne prebrzo.
  Isprobajte strategiju **α_e = ln(e + 1) / (e + 1)**, gde je `e` redni broj
  epizode. Takođe isprobajte **konstantnu stopu** učenja i kroz eksperimente
  nađite adekvatnu vrednost, pa **uporedite brzinu konvergencije** sa
  promenljivom stopom.
- Da biste pratili napredak/konvergenciju, prikažite kako se sa iteracijama `t`
  menjaju **V-vrednosti V_t(s) = max_a Q_t(s, a)**. Uporedite ih sa teorijski
  tačnim vrednostima dobijenim **iteracijom Q-vrednosti**.
- Testirajte konačnu naučenu politiku kroz **10 epizoda** interakcije i
  izračunajte **prosečnu ukupnu nagradu** po epizodi prateći tu politiku. Zatim
  **ponovite za γ = 0.999**. Ima li razlike u odnosu na γ = 0.9 i kako je
  tumačite?

### REINFORCE (zahtevi)

- Ukratko obrazložite **kako ste parametrizovali politiku**.
- Pratite napredak učenja kroz **ukupnu nagradu osvojenu tokom jedne epizode**
  (suma „sirovih", neponderisanih nagrada).
- Grafički prikažite kako se tokom učenja menjaju ta nagrada i **vrednosti
  parametara politike** u neterminalnim stanjima.
- Eksperimentišite sa stopama učenja, kao kod Q-učenja.

### Napomene (predaja)

Python kôd i **izveštaj** sa traženim graficima (PDF ili HTML) predaju se putem
**MS Teams**-a. U izveštaju navedite ili grafički prikažite **konačnu politiku**
koju su algoritmi naučili. Ne zaboravite **Turn In**.

---

## 2. OKRUŽENJE — potpuna specifikacija (za implementaciju)

### Mreža i stanja

```
        col 1     col 2     col 3     col 4     col 5
row A    A1        A2        A3        A4        A5
row B    B1(😰)    B2        B3(😎)    B4        B5(😰)
        term −1            term +1            term −1
```

- **Terminalna stanja (3):** B1 (nagrada −1), B3 (+1), B5 (−1).
- **Neterminalna stanja (7):** A1, A2, A3, A4, A5, B2, B4 → uniformna početna
  raspodela 1/7 svako. (Ovo „7" potvrđuje da B2 i B4 jesu neterminalna.)
- Ukupno **10 stanja**.

### Akcije i susedstva

Akcije: `gore`, `dole`, `levo`, `desno`. Kretanje: `gore/dole` menja red (A↔B)
u istoj koloni; `levo/desno` menja kolonu u istom redu. Pokušaj izlaska iz mreže
= udarac u zid = ostajanje u mestu.

| Stanje | gore | dole | levo | desno |
|--------|------|------|------|-------|
| A1 | zid (A1) | B1 | zid (A1) | A2 |
| A2 | zid (A2) | B2 | A1 | A3 |
| A3 | zid (A3) | B3 | A2 | A4 |
| A4 | zid (A4) | B4 | A3 | A5 |
| A5 | zid (A5) | B5 | A4 | zid (A5) |
| B2 | A2 | zid (B2) | B1 | B3 |
| B4 | A4 | zid (B4) | B3 | B5 |

(B1, B3, B5 terminalna — bez akcija.)

### Model prelaza (stohastika)

Za **izabranu** akciju ishod se računa preko 5 „nameravanih" pomeraja, pa se
udarci u zid spoje u „ostani":

- nameravani smer: **0.6**
- svaki od preostala 3 smera: **0.1**
- „ostani u mestu": **0.1**
- svaki pomeraj koji bi izašao iz mreže → agent **ostaje u istom stanju**.

**Provereni primer (A1, akcija „desno"):** desno→A2 (0.6), gore→zid→A1 (0.1),
dole→B1 (0.1), levo→zid→A1 (0.1), ostani→A1 (0.1) ⟹ **P = {A2: 0.6, A1: 0.3,
B1: 0.1}**. ✔ (poklapa se sa primerom iz teksta domaćeg).

### Nagrade (konvencija je prelazna, R(s, a, s′))

- Prelaz u **B3** ⟶ **+1**; prelaz u **B1** ili **B5** ⟶ **−1** (pa epizoda
  staje).
- Svaki drugi korak (prelaz koji ne ulazi u terminal) ⟶ **−0.04** (živi
  trošak / „living reward", podstiče kratke putanje ka +1).
- γ = 0.9 (osim gde je traženo γ = 0.999 ili γ = 1).

> ⚠️ **Pažnja na konvenciju nagrade.** Teorijski slajdovi pišu nagradu kao
> **R(s)** (zavisi od trenutnog stanja), dok je u ovom domaćem nagrada
> **prelazna R(s, a, s′)** (zavisi od ishoda prelaza). Sve formule ispod su
> napisane sa **r** = stvarno osmotrena nagrada tog koraka, što važi u oba
> slučaja. U implementaciji terminalna nagrada (±1) dobija se „na prelasku" u
> terminal, a ne kao posebna nagrada „u terminalu".

---

## 3. TEORIJA — POZNATI MPO (predavanja P11a uvod + P11b)

### 3.1 Šta je učenje podsticanjem (UP)

- **Učenje sa nadzorom**: za svako stanje (ulaz) `x_i` znamo „tačnu" akciju
  (izlaz) `a_i`.
- **Sekvencijalni problemi**: lako je oceniti konačno stanje, teško pojedinačne
  akcije.
- **UP** ≈ učenje **politike** kroz **pokušaje i greške**, na osnovu nagrada iz
  velikog broja epizoda.
- **Zašto ne planiranje?** Planiranje za dato početno stanje i model vraća
  sekvencu akcija do cilja; u **stohastičnim** okruženjima zahteva česta
  replaniranja i optimalnost nije zagarantovana; **neprimenjivo ako model nije
  dostupan**. Moguće su kombinacije planiranja i UP.

**Mapa kursa:** MPO → (poznat MPO) iteracija vrednosti / Q-vrednosti / politike
→ (nepotpuno poznat MPO) vremenske razlike i bootstrapping = Q-učenje / gradijent
politike = REINFORCE / akter-kritičar.

### 3.2 Markovljev proces odlučivanja (MPO)

MPO = (S, A, P, R, γ):

- **S** — stanja (npr. raspored figura: pozicija, orijentacija, brzina).
- **A** — akcije (dozvoljeni potezi / komande aktuatorima).
- **P_sa(s′)** — verovatnoća prelaska: raspodela narednog stanja s′ ako u s
  primenimo a.
- **R(s)** — nagrada (može zavisiti i od a, s′).
- **γ ∈ [0, 1]** — faktor umanjenja budućih nagrada.

Za sekvencu akcija a0, a1, … sekvenca stanja je s0, s1 ∼ P_{s0,a0}, s2 ∼
P_{s1,a1}, … Ukupna (umanjena) nagrada je:

```
R(s0) + γ R(s1) + γ² R(s2) + …
```

**Zadatak UP:** izborom sekvence akcija maksimizirati **E[ R(s0) + γ R(s1) +
γ² R(s2) + … ]**.

**Politika** (zakon upravljanja, ZU) **π: S → A** preslikava stanja u akcije.

### 3.3 Vrednosna funkcija i Belmanove jednačine

**Vrednosna funkcija politike π:**

```
V^π(s) = E[ R(s0) + γ R(s1) + γ² R(s2) + … | π, s0 = s ]
```

**Belmanove jednačine** (za fiksiran π) — linearan sistem od N_S jednačina sa
N_S nepoznatih:

```
V^π(s) = R(s) + γ Σ_{s′} P_{s,π(s)}(s′) · V^π(s′),  ∀s
```

**Optimalna vrednost stanja (Belmanova optimalnost):**

```
V*(s) = max_a [ R(s) + γ Σ_{s′} P_sa(s′) · V*(s′) ]
```

**Optimalna politika:**

```
π*(s) = arg max_a Σ_{s′} P_sa(s′) · V*(s′)      ...(1)
```

### 3.4 Iteracija vrednosti (Value Iteration)

Za dato R(s), P_sa(s′), γ; |S| = N_S < ∞, |A| = N_A < ∞ — naći V*(s):

```
0. V_0(s) = 0,  ∀s
1. za t = 1, 2, … (do konvergencije):
      V_t(s) = R(s) + max_a γ Σ_{s′} P_sa(s′) · V_{t-1}(s′),  ∀s
2. optimalnu politiku dobiti iz (1)
```

- **Tumačenje:** V_t(s) = optimalna vrednost ako je ostalo `t` koraka. V_0 = 0
  (nema nagrada za 0 koraka), V_1(s) = R(s).
- **Konvergencija:** |V_{t+1}(s) − V_t(s)| ≤ γ^t · max_{s′}|R(s′)| → 0.
  V → V* u svakom slučaju.
- **Sinhrono vs. asinhrono ažuriranje:** sinhrono = nove vrednosti se koriste
  tek u sledećoj iteraciji; asinhrono = čim ažuriramo V(s), odmah ga koristimo
  za ostala stanja (često brže).
- **Kompleksnost jedne iteracije:** O(N_S² · N_A).

### 3.5 Iteracija Q-vrednosti (Q-Value Iteration) — **traži se u domaćem (deo 2)**

**Q-vrednost** = očekivana ukupna nagrada kada u s primenimo a:

```
Q(s, a) = R(s) + γ Σ_{s′} P_sa(s′) · V(s′)
```

**Veza sa V:** za determinističku politiku V^π(s) = Q^π(s, π(s)); za optimalnu
**V*(s) = max_a Q*(s, a)**.

**Algoritam (ono što se implementira):**

```
0. Q_0(s, a) = 0,  ∀(s, a)
1. za t = 1, 2, … (do konvergencije):
      Q_t(s, a) = R(s) + γ Σ_{s′} P_sa(s′) · max_{a′} Q_{t-1}(s′, a′),  ∀(s, a)
```

> Ovo je „tačno rešenje" jer koristi **poznat model** P_sa(s′) (simulator ga
> daje). U domaćem služi kao **etalon (ground truth)** za poređenje sa
> V_t(s) = max_a Q_t(s, a) koje uči Q-učenje.

### 3.6 Iteracija politike (Policy Iteration)

```
0. inicijalizuj π_0 nasumično
1. za t = 0, 1, … (do konvergencije):
      a) (evaluacija) sračunaj V^{π_t}(s), ∀s  — rešavanje N_S × N_S sistema
      b) (poboljšanje) π_{t+1}(s) = arg max_a Σ_{s′} P_sa(s′) · V^{π_t}(s′)
   Posle konačno mnogo iteracija: π_t = π*, V^{π_t} = V*.
```

- Za **male** MPO obično konvergira **brže** od iteracije vrednosti (i kad se
  V malo promeni, optimalna akcija često ostaje ista).
- Za **velike** MPO korak evaluacije je skup: inverzija N_S × N_S matrice je
  O(N_S³).

*(Iteracija politike nije eksplicitno tražena u domaćem, ali je deo gradiva.)*

---

## 4. TEORIJA — NEPOTPUNO POZNAT MPO (predavanje P11c)

Sekcije predavanja: Identifikacija modela · Vremenske razlike i bootstrapping
(Q-učenje) · Istraživanje · Beskonačni prostori · Teorema o gradijentu politike
(REINFORCE) · Akter-kritičar. Ovde su algoritmi koji **ne koriste model** P_sa,
nego uče direktno iz osmotrenih četvorki (s, a, s′, r).

### 4.1 Q-učenje (vremenske razlike, bootstrapping) — **domaći (deo 3)**

Ažuriranje iz svake osmotrene četvorke (s, a, s′, r):

```
Q(s, a) ← (1 − α) Q(s, a) + α [ r + γ · max_{a′} Q(s′, a′) ]
```

ekvivalentno (TD oblik):

```
Q(s, a) ← Q(s, a) + α [ r + γ max_{a′} Q(s′, a′) − Q(s, a) ]
```

- `r` = osmotrena nagrada tog prelaza; ako je s′ **terminalno**,
  max_{a′}Q(s′,a′) = 0 (nema budućnosti).
- **off-policy**: cilj koristi `max_{a′}` bez obzira koja je akcija stvarno
  izabrana. „Bootstrapping" = oslanjanje na tekuću procenu Q(s′, ·).
- **α** = stopa učenja (vidi zahteve domaćeg: α_e = ln(e+1)/(e+1) vs. konstanta).

### 4.2 Istraživanje (ε-gramzivo)

```
a = { slučajna akcija            sa verovatnoćom ε
    { arg max_a Q(s, a)          sa verovatnoćom 1 − ε
```

Balans **istraživanje/eksploatacija**. ε se obično bira eksperimentalno (i može
se smanjivati tokom učenja).

### 4.3 REINFORCE (gradijent politike) — **domaći (deo 4)**

Parametrizovana politika **π_θ(a | s)**. Cilj je maksimizirati očekivanu
nagradu; ažuriranje ide u smeru **skor funkcije** ∇_θ ln π_θ(a | s), ponderisano
povraćajem.

**Povraćaj-do-kraja (return-to-go)** za korak τ, sa nagradama r_t:

```
v_τ = r_τ + γ r_{τ+1} + γ² r_{τ+2} + …   (računa se unazad: v_τ = r_τ + γ v_{τ+1})
```

**Ažuriranje parametara** za svaki korak τ epizode:

```
θ ← θ + α · v_τ · ∇_θ ln π_θ(a_τ | s_τ)
```

**Tipична parametrizacija za kontinualne akcije (Gausova politika)** — po jedan
parametar θ_i (srednja vrednost) za svako stanje S_i, jedinična varijansa:

```
π_θ(a | S_i) = (1 / √(2π)) · exp( −(a − θ_i)² / 2 )
∂/∂θ_j ln π_θ(a | S_i) = (a − θ_j) · δ_ij      (δ_ij = 1 ako j = i, inače 0)
```

> Za **diskretne** akcije (kao u ovom domaćem: gore/dole/levo/desno) prirodnija
> je **softmaks** parametrizacija: π_θ(a|s) = softmax preko skorova θ_{s,a}
> (po parametar za svaki par (s, a) u neterminalnim stanjima). Tada je
> ∇_θ ln π = (indikator izabrane akcije − verovatnoće politike). U izveštaju
> treba „ukratko obrazložiti kako ste parametrizovali politiku" — softmaks po
> (stanje, akcija) je standardan izbor i daje parametre koji se lepo prikazuju
> po neterminalnim stanjima (što domaći traži).

### 4.4 Akter-kritičar (samo pomenuto)

Kombinuje učenje politike (**akter**, REINFORCE-stil) i učenje vrednosne funkcije
(**kritičar**, TD-stil); kritičar daje baznu liniju koja smanjuje varijansu
ažuriranja. *(Nije tražen u domaćem.)*

---

## 5. ČETIRI ZADATKA DOMAĆEG — sažeto, sa formulama i pseudokodom

### (1) Simulator
Drži interno stanje. Na ulaz `step(action)`:
- sempluj naredno stanje po modelu prelaza iz §2 (0.6 / 0.1×3 / 0.1, zidovi →
  ostani),
- vrati `(reward, next_state, done)`,
- terminali B1/B3/B5 → done = True, nagrada ±1 na ulasku.
- Dodatno izloži `transition_dist(s, a) → {s′: P}` (potrebno za iteraciju
  Q-vrednosti). Korisno: `reset()` bira uniformno jedno od 7 neterminalnih
  stanja.

### (2) Iteracija Q-vrednosti (koristi model)
```
Q[s,a] = 0
ponavljaj do konvergencije:
    za svako (s, a):
        Q[s,a] = Σ_{s′} P(s′|s,a) · ( r(s,a,s′) + γ · max_{a′} Q[s′,a′] )
V*[s] = max_a Q[s,a]    # etalon za poređenje sa Q-učenjem
```
(Pošto je nagrada prelazna, r ulazi unutar sume; za terminalne s′ je
max_{a′}Q[s′,·] = 0.)

### (3) Q-učenje (bez modela, ε-gramzivo)
```
Q[s,a] = 0
za epizodu e = 1, 2, …:
    s = reset()
    dok nije done:
        a = ε-greedy(Q, s, ε)
        (r, s′, done) = step(a)
        target = r + (0 ako done inače γ · max_{a′} Q[s′,a′])
        Q[s,a] += α_e · (target − Q[s,a])
        s = s′
prati V_t(s) = max_a Q_t(s,a); uporedi sa (2)
```
Eksperimenti: α_e = ln(e+1)/(e+1) vs. konstantno α; ε kroz eksperimente; test
10 epizoda → prosečna ukupna nagrada; ponoviti za γ = 0.999 i prokomentarisati.

### (4) REINFORCE (bez modela, gradijent politike)
```
θ = 0  (npr. softmaks logit po (s, a) za 7 neterminalnih stanja)
za epizodu e = 1, 2, …:
    generiši epizodu (s_t, a_t, r_t) prateći π_θ
    računaj v_τ unazad: v_τ = r_τ + γ v_{τ+1}
    za svaki korak τ:
        θ += α_e · v_τ · ∇_θ ln π_θ(a_τ | s_τ)
prati: sumu sirovih nagrada po epizodi + vrednosti parametara po neterminalnim
       stanjima (grafici)
```

---

## 6. REŠENI/POSTAVLJENI ZADACI SA VEŽBI (Vežbe 6) — reference

> Metod ovih zadataka direktno odgovara delovima 3 i 4 domaćeg. Strelice
> (↑ ↓ ← →) su rekonstruisane iz slika.

### Q-učenje — Zadatak 1
Stanja A, B, C, D; **B i D terminalna**. U A: akcije {→, ↓}. U C: akcije {↑, →}.
α = 0.6, γ = 0.9. Četvorke (s, a, s′, r):
- Ep1: (A, →, B, −1), (B, ·, ·, −3)
- Ep2: (A, ↓, C, −1), (C, →, D, −1), (D, ·, ·, 7)
- Ep3: (A, →, B, −1), (B, ·, ·, −3)
- Ep4: (A, ↓, C, −1), (C, ↑, A, −1), (A, →, B, −1), (B, ·, ·, −3)

Traži se: naučene Q-vrednosti posle 4 epizode + optimalne akcije u neterminalnim
stanjima. **Metod:** primeni Q(s,a) ← (1−α)Q(s,a) + α[r + γ max_{a′}Q(s′,a′)]
**redom po vremenu**; Q-vrednosti se dele između epizoda; terminalna „·"
četvorka daje terminalnu nagradu (max budućeg = 0).

### Q-učenje — Zadatak 2
Neterminalna 0 (početno), 1, 2, 3, 4; terminalno T. Akcije N (nastavi) i K
(kraj); K vodi u T; ako N, naredno stanje bira stohastičko okruženje. Nenulta
nagrada samo kada je s′ = T. α = 0.1, γ = 1.
- Ep1: (0, N, 3, 0), (3, N, 4, 0), (4, K, T, 4)
- Ep2: (0, N, 1, 0), (1, N, 4, 0), (4, N, T, −1)

Traži se: šta bi Q-učenje naučilo iz ovih opservacija.

### REINFORCE — Zadatak 1
S ∈ {S0, S1, S2}, akcije a ∈ ℝ. α = 0.1, γ = 0.5. Epizoda:
(S0, 0.5, S1, −1), (S1, −1, S2, −1), (S2, 0, ·, 4).

**Rešeno (Gausova politika, θ = (θ0, θ1, θ2), početno 0):**
- Povraćaji unazad: **v3 = 4**, **v2 = −1 + 0.5·4 = 1**, **v1 = −1 + 0.5·1 = −0.5**.
- Skor: ∂/∂θ_i ln π = (a − θ_i).
- θ0 ← 0 + 0.1·v1·(0.5 − 0) = 0.1·(−0.5)·0.5 = **−0.025**
- θ1 ← 0 + 0.1·v2·(−1 − 0) = 0.1·1·(−1) = **−0.1**
- θ2 ← 0 + 0.1·v3·(0 − 0) = **0**
- Konačno **θ = (−0.025, −0.1, 0)**.

### REINFORCE — Zadatak 2
Mreža: gornji red `1 2 3 4`, ispod stanja 2 je `5`. **1, 4, 5 terminalna**
(1 → −1, 4 → +3, 5 → −3). U stanju 2: akcije {←, →, ↓}; u stanju 3: {←, →}.
- a) predložiti parametrizaciju politike (npr. softmaks po dozvoljenim akcijama);
- b) parametri posle 3 epizode:
  - Ep1: (2, →, 3, 0), (3, →, 4, 0), (4, ·, ·, +3)
  - Ep2: (2, ←, 1, 0), (1, ·, ·, −1)
  - Ep3: (2, ↓, 5, 0), (5, ·, ·, −3)

### REINFORCE — Za razmišljanje
Auto se kreće pravolinijski ka cilju; poznati su udaljenost `d` i brzina `v`;
diskretne akcije {koči, ne radi ništa, daj gas}. Predložiti parametrizaciju
politike, pa iz nje izvesti izraz za **skor** i pravila **ažuriranja parametara**
(poznata „zarada" v_t po koraku, kao i α i γ). *(Vežba za softmaks politiku nad
linearnim/feature skorovima funkcije od d i v.)*

### Rešeni ispitni primer Q-učenja (iz predavanja P11c)
S ∈ {A,B,C,D}, akcije {←,↑,→,↓}; α = 0.5, γ = 0.8. Epizode:
(A,→,B,0),(B,·,·,10) i (A,↓,C,0),(C,→,D,0),(D,·,·,−4).
Rešenje: Q(s,a) ← (1−α)Q(s,a) + α[r + γ max_{a′}Q(s′,a′)] ⟹
**Q(B,·) = 0.5·10 = 5**, **Q(D,·) = 0.5·(−4) = −2** (terminali; budući max = 0).

---

## 7. Mapa: koji izvor pokriva šta

| Tema | Izvor |
|------|-------|
| Uvod u UP, nadzor vs. podsticanje, zašto ne planiranje, mapa kursa | `P11a…uvod` (8 slajdova) |
| MPO, V/Q, Belman, iteracija vrednosti / Q-vrednosti / politike | `P11b…poznati-mpo` (62 slajda, slikovni; CS188 gridworld) |
| Q-učenje, istraživanje, REINFORCE, akter-kritičar (formule iz rešenih primera) | `P11c…nepoznati-mpo` (naslovi + 2 rešena zadatka) |
| Rešeni/postavljeni zadaci Q-učenja i REINFORCE | `Vežbe_6…` (5 strana) |
| Tekst domaćeg + okruženje | `vi26dom3` (3 strane) |

**Ključne formule na jednom mestu**
- Belman (π): V^π(s) = R(s) + γ Σ_{s′} P_{s,π(s)}(s′) V^π(s′)
- Belman (optimalno): V*(s) = max_a [ R(s) + γ Σ_{s′} P_sa(s′) V*(s′) ]
- Iteracija vrednosti: V_t(s) = R(s) + max_a γ Σ_{s′} P_sa(s′) V_{t−1}(s′)
- Iteracija Q-vrednosti: Q_t(s,a) = R(s) + γ Σ_{s′} P_sa(s′) max_{a′} Q_{t−1}(s′,a′)
- Q-učenje: Q(s,a) ← Q(s,a) + α[ r + γ max_{a′} Q(s′,a′) − Q(s,a) ]
- REINFORCE: θ ← θ + α v_τ ∇_θ ln π_θ(a_τ|s_τ),  v_τ = r_τ + γ v_{τ+1}
