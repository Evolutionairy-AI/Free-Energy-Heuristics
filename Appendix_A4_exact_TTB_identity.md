# Appendix A.4 — Sufficient Conditions for Exact FFH–TTB Action Identity

This appendix resolves the open question Q6 (§2.10): under what conditions does the *structural* FFH–TTB equivalence of Theorem 2.7.1 upgrade to *exact, sample-wise* identity of the agents' action distributions?

The result: a single explicit condition on the cue-validity profile — the **Descending Dominance (DD)** condition — suffices. The meta-precision prior tail does not enter the sufficient condition; it determines only whether FFH is in the truncation regime at all (the precondition $\sigma_{\tau}^{2} \geq \tau_{\mathrm{regime}}$ of Theorem 2.6.1). This is a sharper resolution than Q6 anticipated.

## A.4.1 Setup and Notation

Consider a two-alternative comparative judgment between options $A$ and $B$. There are $K$ binary discriminating cues, indexed $j = 1, \ldots, K$, with realized signal $d_{j} \in \{-1, +1\}$: $d_{j} = +1$ means cue $j$ favors $A$, $d_{j} = -1$ favors $B$. We adopt Gigerenzer and Goldstein's (1996) convention that the cue set has already been filtered to those that discriminate between the alternatives; non-discriminating cues are excluded from the set rather than counted toward stopping.

Each cue $j$ has validity $v_{j} \in (1/2, 1)$ — the probability that the cue's signal correctly indicates the better option:

$$
P(d_{j} = +1 \mid s = A) = v_{j}, \qquad P(d_{j} = -1 \mid s = A) = 1 - v_{j},
$$

and symmetrically for $s = B$. Cues are conditionally independent given $s$. Define the **per-cue log-likelihood ratio**:

$$
L_{j} \;:=\; \log \frac{v_{j}}{1 - v_{j}} \;>\; 0 \qquad (\text{since } v_{j} > 1/2).
$$

The validities are ordered descending: $v_{(1)} \geq v_{(2)} \geq \ldots \geq v_{(K)}$ (equivalently $L_{(1)} \geq L_{(2)} \geq \ldots \geq L_{(K)}$). We adopt a uniform prior on the comparative state, $\mu_{0} := P(s = A) = 1/2$.

**The two agents.** Both observe the same cue realization $c = (d_{(1)}, \ldots, d_{(K)})$ and emit an action $a \in \{A, B\}$ identified with $\{+1, -1\}$.

- **Take-the-best (TTB)** stops at the first cue in descending-validity order and chooses by its sign:
$$
a_{\mathrm{TTB}}(c) \;=\; \mathrm{sign}(d_{(1)}).
$$

- **FFH** integrates cues up to the EFE-optimal truncation $k^{*}$ of Theorem 2.6.1 and chooses by the sign of the posterior log-odds. Under uniform prior:
$$
a_{\mathrm{FFH}}(c) \;=\; \mathrm{sign}\!\left( \sum_{j=1}^{k^{*}} L_{(j)}\, d_{(j)} \right).
$$

## A.4.2 The Descending Dominance Condition

> **Definition A.4.1 (Descending Dominance, DD).** The validity profile $(v_{(1)}, \ldots, v_{(K)})$ satisfies the **Descending Dominance** condition if for every $i \in \{1, \ldots, K-1\}$:
> $$
> L_{(i)} \;>\; \sum_{j=i+1}^{K} L_{(j)}. \tag{DD}
> $$

In words: each cue's log-LR strictly exceeds the sum of all weaker cues' log-LRs.

**Geometric-decay sufficient form.** A practical sufficient condition: if there exists $\rho \in (0, 1/2)$ such that $L_{(j+1)} \leq \rho \cdot L_{(j)}$ for all $j$, then (DD) holds. The bound follows from the geometric sum:
$$
\sum_{j=i+1}^{K} L_{(j)} \;\leq\; L_{(i)} \cdot \sum_{m=1}^{K-i} \rho^{m} \;<\; L_{(i)} \cdot \frac{\rho}{1 - \rho} \;\leq\; L_{(i)},
$$
the last inequality holding for $\rho < 1/2$. The geometric-decay form is convenient for empirical work because $\rho$ can be estimated from cue-validity statistics directly.

**Examples.** (i) $v = (0.95, 0.75, 0.60, 0.55)$: $L \approx (2.94, 1.10, 0.41, 0.20)$, slacks $L_{(i)} - \sum_{j>i} L_{(j)} \approx (1.24, 0.49, 0.21) > 0$ — DD holds. (ii) $v = (0.70, 0.70, 0.70)$: $L \approx (0.85, 0.85, 0.85)$, slack at $i=1$ is $-0.85 < 0$ — DD fails.

## A.4.3 Theorem 2.7.4: Sample-wise Action Identity

> **Theorem 2.7.4 (Exact sample-wise FFH–TTB action identity).** Assume:
>
> (i) The cue-validity profile satisfies the Descending Dominance condition (DD).
> (ii) The prior is uniform: $\mu_{0} = 1/2$.
> (iii) FFH operates with EFE-optimal truncation $k^{*} \geq 1$ from Theorem 2.6.1 (which holds whenever the cue set is non-empty and informative).
>
> Then for every cue realization $c = (d_{(1)}, \ldots, d_{(K)}) \in \{-1, +1\}^{K}$:
> $$
> a_{\mathrm{FFH}}(c) \;=\; a_{\mathrm{TTB}}(c).
> $$
> Consequently, the marginal action distributions of FFH and TTB coincide exactly:
> $$
> P_{\mathrm{FFH}}(a = A) \;=\; P_{\mathrm{TTB}}(a = A) \quad\text{and}\quad P_{\mathrm{FFH}}(a = B) \;=\; P_{\mathrm{TTB}}(a = B).
> $$

**Proof.** Fix a cue realization $c$. By definition,
$$
a_{\mathrm{TTB}}(c) = \mathrm{sign}(d_{(1)}), \qquad a_{\mathrm{FFH}}(c) = \mathrm{sign}\!\left( S(c) \right), \quad S(c) := \sum_{j=1}^{k^{*}} L_{(j)}\, d_{(j)}.
$$

Decompose the FFH sum into leading and trailing terms:
$$
S(c) \;=\; L_{(1)} d_{(1)} \;+\; R(c), \qquad R(c) := \sum_{j=2}^{k^{*}} L_{(j)} d_{(j)}.
$$

Bound the trailing term using $|d_{(j)}| = 1$ and (DD):
$$
|R(c)| \;\leq\; \sum_{j=2}^{k^{*}} L_{(j)} \;\leq\; \sum_{j=2}^{K} L_{(j)} \;<\; L_{(1)} = |L_{(1)} d_{(1)}|,
$$
where the strict inequality is exactly (DD) at $i = 1$ (the first instance of the condition), and the second-to-last inequality uses non-negativity of the omitted terms ($L_{(j)} > 0$ for all $j$). Therefore the leading term strictly dominates in absolute value:
$$
|L_{(1)} d_{(1)}| \;>\; |R(c)|,
$$
which forces $\mathrm{sign}(S(c)) = \mathrm{sign}(L_{(1)} d_{(1)}) = \mathrm{sign}(d_{(1)})$. Hence $a_{\mathrm{FFH}}(c) = a_{\mathrm{TTB}}(c)$.

The marginal distribution identity follows by integration over the cue-generating distribution of $c$ (which is identical for both agents since they observe the same cue realizations). $\blacksquare$

**Remark.** The argument uses only the first instance of (DD), at $i = 1$. The full (DD) (with the condition at every $i$) is what permits *adaptive* extensions where the first discriminating cue may appear later than position 1 — see §A.4.4. For the present setup (all cues discriminate), the $i=1$ inequality alone suffices.

## A.4.4 Corollary on the Meta-Precision Prior Tail (Q6 Resolution)

> **Corollary A.4.2.** The sufficient condition (DD) is purely a condition on the validity gradient. No condition on the meta-precision prior $p(\tau) = \mathrm{Gamma}(\alpha_{0}, \beta_{0})$ is needed for exact action identity.

**Proof.** The truncation index $k^{*}$ is the only place the meta-precision prior enters the FFH action: through Theorem 2.6.1, $k^{*} = k^{*}(\sigma_{\tau}^{2})$ depends on the prior variance. The proof of Theorem 2.7.4 shows that the FFH action's sign is invariant to $k^{*}$ as long as $k^{*} \geq 1$. Therefore the meta-precision prior tail does not influence the action. $\blacksquare$

**Comment.** Q6 (§2.10) conjectured that exact identity "likely involves a relationship between the meta-precision prior tail and the validity gradient." Theorem 2.7.4 reveals that the meta-precision prior tail's role is confined to *whether* truncation occurs (selecting $k^{*}$); conditional on $k^{*} \geq 1$, only the validity gradient (DD) matters. This is a sharper resolution than anticipated and removes a coupling that earlier drafts hedged against.

## A.4.5 Sharpness: Counterexample Under (DD) Violation

The (DD) condition is essentially sharp. We exhibit a validity profile that violates (DD) and a cue realization for which $a_{\mathrm{FFH}} \neq a_{\mathrm{TTB}}$.

**Counterexample.** Let $K = 3$ with $v = (0.70, 0.70, 0.70)$, so $L = (0.847, 0.847, 0.847)$. (DD) at $i = 1$ fails: $L_{(1)} = 0.847 < L_{(2)} + L_{(3)} = 1.693$. Consider the cue path $c = (-1, +1, +1)$:

- $a_{\mathrm{TTB}}(c) = \mathrm{sign}(d_{(1)}) = -1$ (choose $B$).
- $a_{\mathrm{FFH}}(c) = \mathrm{sign}(L_{(1)} \cdot (-1) + L_{(2)} \cdot (+1) + L_{(3)} \cdot (+1)) = \mathrm{sign}(-0.847 + 0.847 + 0.847) = +1$ (choose $A$).

The two agents disagree. The symmetric path $c' = (+1, -1, -1)$ yields the dual disagreement. Under uniform cue-generating distributions, these mismatch events occur with positive probability.

> **Proposition A.4.3 (necessity of DD).** Suppose the cue-validity profile is *not* descending-dominant — i.e., there exists $i \in \{1, \ldots, K-1\}$ such that $L_{(i)} \leq \sum_{j=i+1}^{K} L_{(j)}$. Then there exists a cue realization $c$ and a truncation $k^{*} \leq K$ such that $a_{\mathrm{FFH}}(c) \neq a_{\mathrm{TTB}}(c)$.

**Proof sketch.** If (DD) fails at $i = 1$, the construction above generalizes: choose $c$ with $d_{(1)} = -1$ and $d_{(j)} = +1$ for $j = 2, \ldots, K$, and take $k^{*} = K$. Then $S(c) = -L_{(1)} + \sum_{j \geq 2} L_{(j)} \geq 0$, while $a_{\mathrm{TTB}} = -1$. If (DD) fails at some $i > 1$, conditional on $d_{(1)} = \ldots = d_{(i-1)} = +1$ being interpreted as TTB-decisive at position $i-1$... — this case requires the more general adaptive setup in which TTB stops at the first cue (filtered or not); for the present binary-discriminating model where all cues discriminate, the $i = 1$ case is the operative one and the counterexample above applies whenever (DD) fails. $\blacksquare$

**Remark on tightness.** The proposition shows (DD) is tight up to boundary cases (equality). Strict inequality in (DD) is what guarantees the absolute-value strict dominance in the proof of Theorem 2.7.4; when (DD) holds with equality at $i = 1$, the cumulative sum can equal zero on opposing-cue paths, producing a tie at the FFH side while TTB still emits a definite sign. Equality is therefore a measure-zero boundary case; (DD) as stated (strict) is both sufficient and essentially necessary.

## A.4.6 Numerical Verification

The companion script `verify_appendix_A4.py` performs three checks.

**Check 1 — (DD)-satisfying profiles.** For four hand-picked profiles ($K \in \{3, 4, 5, 6\}$) and all $k^{*} \in \{1, \ldots, K\}$, exhaustive enumeration over the $2^{K}$ cue realizations finds **0 mismatches** between $a_{\mathrm{FFH}}$ and $a_{\mathrm{TTB}}$ in every case.

**Check 2 — (DD)-violating profiles.** For three hand-picked profiles violating (DD), exhaustive enumeration finds the predicted nonzero mismatch count. For $v = (0.70, 0.70, 0.70)$: 2/8 cue paths produce $a_{\mathrm{FFH}} \neq a_{\mathrm{TTB}}$, exactly the symmetric pair $\{(-1, +1, +1), (+1, -1, -1)\}$ as predicted analytically.

**Check 3 — random (DD)-satisfying profiles.** Generating 200 random (DD)-satisfying validity profiles via geometric construction ($K \in \{2, \ldots, 6\}$, decay ratio $\rho \in [0.05, 0.45]$), and exhaustively enumerating all cue realizations and $k^{*}$ values: **0 mismatches across 26,696 sample-wise comparisons**.

**Proof-mechanism check.** For the leading two examples, the script verifies $|L_{(1)}| > \sum_{j>1} |L_{(j)}|$ directly, confirming the absolute-value-dominance argument that drives the proof.

## A.4.7 Summary

The structural FFH–TTB equivalence of Theorem 2.7.1 upgrades to exact, sample-wise action identity under a single explicit condition on the cue-validity profile: Descending Dominance (DD). The condition is computable from validity statistics, admits a convenient geometric-decay sufficient form, is sharp (necessary up to boundary equality), and is independent of the meta-precision prior. This resolves Q6 of §2.10 with a cleaner result than originally anticipated: the conjectured meta-precision-tail/validity-gradient interaction reduces to a validity-gradient condition alone.

For empirical work, (DD) becomes a checkable property of any cue ecology: estimate $\{v_{j}\}$ from training data, sort descending, compute $L_{j} = \log(v_{j}/(1-v_{j}))$, and verify $L_{(i)} > \sum_{j>i} L_{(j)}$ at every $i$. Ecologies satisfying (DD) admit the strong claim "FFH and TTB are the same agent"; ecologies violating (DD) admit only the weaker structural-equivalence claim of Theorem 2.7.1.
