# mlBased-NIDS-research
# Research Paper — Key Points & Reasoning
> All main arguments, findings, and reasoning for the paper.
> Course: Network Protocols & Security (NPS)

---

## Paper Title (Draft)

"Dataset Shift in ML-Based Intrusion Detection Systems:
A Protocol-Level Analysis of Generalization Failure"

---

## Core Argument (One Paragraph)

ML-based Intrusion Detection Systems trained on historical network datasets
achieve high performance within their training distribution but fail
significantly when deployed in different network environments. This failure
is not random — it maps directly to protocol-level behavioral differences
between attack generations, structural differences in how datasets were
collected, and the evolution of attack techniques between 2003 and 2017.
Even protocol-unaware features (volume, timing, intensity) that should
theoretically be stable across environments exhibit significant distribution
shift, suggesting that dataset shift in IDS is a fundamental and largely
unsolved problem.

---

## Research Question

> "Do ML models trained on NSL-KDD generalize to CICIDS 2017,
>  and can the performance gap be explained at the network protocol level?"

---

## Datasets

### NSL-KDD
- Collected: 1999, refined 2009
- Environment: Simulated lab network
- Collection method: Per-CONNECTION features, manually engineered
- Features: 41, semantic (flag states, service names, error rates)
- Labels: normal, DoS, Probe, R2L, U2R
- Size: ~125,000 training rows

### CICIDS 2017
- Collected: 2017
- Environment: Realistic modern network (Canadian Institute for Cybersecurity)
- Collection method: Per-FLOW features, CICFlowMeter automated extraction
- Features: 78, statistical (packet counts, byte totals, rates, window sizes)
- Labels: BENIGN + 14 modern attack types
- Size: ~2.8 million rows (sampled to 10% for experiments)

### The Fundamental Collection Difference
```
NSL-KDD  → one row = one complete CONNECTION
            features describe the entire conversation semantically

CICIDS   → one row = one FLOW (bidirectional packet stream)
            features describe statistical properties of the flow
```
This difference in collection philosophy means the same attack
generates completely different feature signatures in each dataset.

---

## Feature Set Used — Protocol-Unaware Control Group

6 features chosen deliberately because they require NO protocol
grammar knowledge to measure:

| Feature | Definition | KDD Source | CICIDS Source |
|---|---|---|---|
| `duration` | Time elapsed | `duration` (seconds) | `Flow Duration` ÷ 1,000,000 |
| `src_bytes` | Forward data volume | `src_bytes` | `Total Length of Fwd Packets` |
| `dst_bytes` | Backward data volume | `dst_bytes` | `Total Length of Bwd Packets` |
| `bytes_per_sec` | Data rate | computed | computed |
| `byte_ratio` | Direction asymmetry | src/(dst+1) | src/(dst+1) |
| `protocol_type` | Transport protocol | `protocol_type` | one-hot encoded |

### Why Protocol-Unaware Features Matter
These features should be the MOST stable across datasets because
they don't depend on protocol behavior that changes between attack
generations. If even these features show massive shift, it proves
the problem is fundamental — not just a protocol grammar mismatch.

### Feature Mapping Quality
```
duration      → EXCELLENT (unit conversion only)
protocol_type → GOOD (value mapping only)
src_bytes     → GOOD (actual bytes in both after correction)
dst_bytes     → GOOD (actual bytes in both after correction)
bytes_per_sec → MODERATE (computed, sensitive to duration=0 edge cases)
byte_ratio    → MODERATE (proxy for traffic direction asymmetry)
```

### Dropped Features & Why
```
count        → NSL-KDD: cross-connection window count
               CICIDS: no equivalent (per-flow only, no cross-flow view)
               Reason: Structural impossibility — CICFlowMeter cannot
               observe cross-flow behavior by design

flag         → NSL-KDD: TCP state categorical (SF, S0, REJ, RSTO)
               CICIDS: Fwd Header Length (continuous bytes)
               Reason: Conceptually incompatible — categorical state
               vs continuous measurement

serror_rate  → NSL-KDD: SYN error rate over 100-connection window
               CICIDS: Init_Win_bytes_backward (single packet field)
               Reason: Window-based rate vs single observation
```

### ICMP Protocol Note
CICIDS 2017 contains no ICMP traffic in the sampled dataset.
Rather than dropping ICMP from NSL-KDD or adding fake zero columns,
ICMP is retained as a one-hot encoded feature.
Its absence in CICIDS is itself evidence of dataset shift —
modern network environments handle ICMP differently than 2003 lab setups.

---

## Protocol-Level Analysis — Why Attacks Look Different

### DoS Attacks
```
NSL-KDD (2003 era):
  Mechanism: Simple SYN flood
  TCP behavior: SYN sent, no ACK (half-open connections)
  Feature signature:
    duration  = 0 (connection never completes)
    src_bytes = 0 (SYN has no payload)
    dst_bytes = 0 (server never responds)
    flag      = S0 (SYN error)
    serror_rate = 1.0 (100% SYN errors)

CICIDS 2017 (modern):
  Mechanisms: DoS Hulk, GoldenEye, Slowloris, Slowhttptest
  TCP behavior: COMPLETE connections (HTTP-based floods)
               or slow connection exhaustion
  Feature signature:
    duration  = long (connections kept alive deliberately)
    src_bytes = large (HTTP requests sent)
    dst_bytes = large (server responds before being overwhelmed)
    → looks like normal heavy traffic to a model trained on SYN floods
```

### Probe / Reconnaissance
```
NSL-KDD:
  Sequential port scanning (port 1,2,3...)
  High serror_rate (RST responses from closed ports)
  Very short duration per connection

CICIDS 2017:
  Randomized nmap scanning (evades sequential detection)
  Slower scan rates (timing delays between probes)
  OS fingerprinting mixed in
  Different timing patterns entirely
```

### R2L (Remote to Local)
```
NSL-KDD:
  Rapid brute force — many failed logins quickly
  High num_failed_logins, sudden logged_in=1

CICIDS 2017:
  FTP-Patator, SSH-Patator — credential stuffing
  Timing delays between attempts to evade rate limiting
  Web application attacks (SQL injection, XSS) over HTTP/HTTPS
  Completely different protocol behavior
```

### U2R (User to Root)
```
NSL-KDD:
  Classic buffer overflows
  root_shell flips 0→1, high num_root

CICIDS 2017:
  Modern kernel exploits
  Web shell uploads over HTTPS (encrypted — harder to detect)
  Different feature signatures entirely
```

---

## Three Types of Dataset Shift Present

### 1. Covariate Shift
Same feature name, different value distributions:
```
duration (normal traffic):
  KDD:    0-100 seconds
  CICIDS: 0-3600 seconds (modern apps keep connections open longer)

src_bytes (DoS attack):
  KDD:    0 bytes (SYN flood has no payload)
  CICIDS: large values (HTTP floods send real data)
```

### 2. Prior Probability Shift
Attack class balance differs between datasets:
```
KDD:    DoS ~50% of attacks
CICIDS: DoS ~30% of attacks, many new categories (Bot, Infiltration, etc.)
Model's internal thresholds calibrated for KDD proportions
```

### 3. Concept Drift
Same attack label, fundamentally different behavior:
```
"DoS" in KDD   = SYN flood (incomplete connections)
"DoS" in CICIDS = HTTP flood (complete connections)
Same label. Opposite feature signatures.
Model learned the wrong concept of "DoS".
```

---

## Structural Collection Difference (Key Finding)

Beyond value distributions, the datasets differ in what they can observe:

```
NSL-KDD count feature:
  "How many connections targeted the same host in last 2 seconds?"
  → requires observing MULTIPLE connections simultaneously
  → network-wide view

CICFlowMeter:
  Operates per-flow — cannot observe cross-flow patterns
  No equivalent to count feature exists
  → single-flow view only

This is not a mapping problem. It is a fundamental observability gap.
A model trained on NSL-KDD's count feature learned network-wide
attack intensity patterns that simply cannot be represented
in CICIDS's per-flow feature space.
```

---

## Experimental Results — Logistic Regression (Control Features)

### Phase 1 — Within Dataset (KDD train → KDD test)
```
Class 0 (normal):  Precision 0.64  Recall 0.97  F1 0.78
Class 1 (attack):  Precision 0.97  Recall 0.59  F1 0.74
Overall F1: 0.7363
```

### Phase 2 — Cross Dataset (KDD train → CICIDS test)
```
Class 0 (normal):  Precision 0.81  Recall 0.77  F1 0.79
Class 1 (attack):  Precision 0.21  Recall 0.25  F1 0.23
Overall F1: 0.2312
```

### Summary
```
Within-dataset F1:  0.7363
Cross-dataset F1:   0.2312
Performance drop:   0.5051  (50 point drop)
```

### Key Observations From Results

**Observation 1 — Asymmetric Failure**
```
Normal traffic detection:  relatively stable (0.78 → 0.79)
Attack detection:          complete collapse (0.74 → 0.23)
The shift specifically destroys attack recognition
not general traffic understanding
```

**Observation 2 — Precision-Recall Inversion**
```
On KDD:    high precision (0.97), low recall (0.59)
           model learned to be conservative — only flag obvious attacks

On CICIDS: low precision (0.21), low recall (0.25)
           modern attacks don't look "obvious" by KDD standards
           model flags randomly AND misses most attacks
```

**Observation 3 — Protocol-Unaware Features Still Shift**
```
These 6 features require no protocol grammar knowledge
Yet they still produce a 50-point F1 drop
This proves the shift is not just about protocol complexity
Even the most basic measurements (bytes, time, rate) are
non-transferable across 14 years of network evolution
```

**Observation 4 — Within-Dataset F1 Lower Than Expected**
```
Expected: 0.90+ for LogReg on KDD
Actual:   0.73
Reason:   Protocol-unaware features alone are insufficient
          to fully capture attack patterns even within KDD
          The dropped protocol-aware features (flags, error rates)
          carry significant discriminative power
→ Sets up the next experiment: adding protocol-aware features
  should raise within-dataset F1 but cause even more cross-dataset drop
```

---

## Pending Experiments

### Experiment 2 — Random Forest (Control Features)
Expected: Higher within-dataset F1 than LogReg
Expected: Worse cross-dataset drop than LogReg (memorizes thresholds)

### Experiment 3 — SVM (Control Features)
Expected: Similar within-dataset F1 to LogReg
Expected: Most stable cross-dataset performance (margin-based)

### Experiment 4 — All Models (Protocol-Aware Features Added)
Expected: Higher within-dataset F1 across all models
Expected: Larger cross-dataset drop — protocol features are more
          dataset-specific than volume/timing features

### Experiment 5 — Mitigation Strategies
```
Mitigation A: Feature normalization (already applied via StandardScaler)
Mitigation B: Use only top N features by importance
Mitigation C: Threshold tuning (adjust decision boundary)
For each: measure impact on cross-dataset F1
Goal: show what helps, what doesn't, what remains unsolved
```

---

## Paper Argument Structure

```
Section 1 — Introduction
  The IDS accuracy illusion: 99% on benchmark ≠ reliable in production

Section 2 — Background
  NSL-KDD vs CICIDS: collection methodology differences
  Three types of dataset shift
  Why protocol evolution matters

Section 3 — Feature Analysis
  Protocol-unaware feature selection rationale
  Feature mapping quality assessment
  Structural observability gap (count feature)

Section 4 — Experiments
  Logistic Regression results
  Random Forest results
  SVM results
  Model comparison table

Section 5 — Protocol-Level Analysis
  Per-attack-type explanation of shift
  DoS evolution: SYN flood → HTTP flood
  Why features that worked in 2003 fail in 2017

Section 6 — Mitigation Analysis
  What helps, what doesn't, what remains unsolved

Section 7 — Conclusion
  Protocol-unaware features shift just as much as protocol-aware ones
  Fundamental observability gaps cannot be resolved by feature mapping
  Dataset shift in IDS is unsolved — evaluation methodology must change
```

---

## Strongest Sentences For The Paper

> "A 50-point F1 drop using only protocol-unaware features demonstrates
>  that dataset shift in IDS is not merely a protocol grammar mismatch —
>  it reflects 14 years of fundamental change in how attacks are conducted
>  and how networks operate."

> "The model's failure is asymmetric: normal traffic detection remains
>  relatively stable while attack detection collapses, suggesting the model
>  learned a 2003-specific definition of anomaly that no longer applies."

> "CICFlowMeter's per-flow collection methodology creates a structural
>  observability gap — cross-flow behavioral features present in NSL-KDD
>  cannot be represented in CICIDS regardless of feature engineering effort."

> "High in-distribution accuracy is not a reliable indicator of
>  real-world IDS robustness. A model achieving 0.97 precision on
>  known attack patterns may simultaneously miss 75% of attacks
>  in a deployment environment collected 14 years later."

---

*Updated after: Logistic Regression control feature experiment*
*Next update: After Random Forest and SVM results*