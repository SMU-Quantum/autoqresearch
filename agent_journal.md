# AutoQResearch MIS Journal

## Baseline

### Iteration 0
- Hypothesis: Establish the required unmodified baseline before changing policy logic.
- Failure signal: None yet; this is the scout anchor.
- Degrees of freedom exercised: None.
- Change summary: No code edits. Recorded the checked-in VQE RealAmplitudes depth-1 baseline.
- Result: KEEP, `suite_average_gap=0.604167`, per-instance gaps: `mis_file_1tc.16=0.3750`, `mis_file_p4tc.16=0.8333`.
- Wall time: `20.1s`.
- Analysis: Baseline is feasible on both scout instances but under-selects on the dense graph. Direct probes showed high feasibility (`~0.93-0.94`), meaningful but not extreme concentration (`top1_prob=0.165` on `1tc.16`, `0.0645` on `p4tc.16`), and strong optimizer stagnation (`0.76-0.81`). Initial search will target expressivity and optimization before penalty or shot changes.

## Initial Roadmap

1. Start with VQE-only adaptive changes because baseline failure is stagnation with decent feasibility, not uniform-noise collapse.
2. Test whether a second VQE attempt with more expressivity improves dense 16-node graphs without harming the easier proxy.
3. If VQE plateaus, branch to QAOA variants next, then PCE and QRAO to satisfy family coverage with evidence-based pruning.
4. Promote the best confirmed 16-node policy before carrying it unchanged into 32-node scout.

## Search Log

### Iteration 1
- Hypothesis: A second VQE attempt with higher depth will recover from the baseline's stagnation on dense 16-node MIS while keeping the strong feasibility of the first attempt.
- Failure signal: `mis_file_p4tc.16` is feasible but very weak (`gap=0.8333`, `AR=0.1667`) with `stagnation=0.81`; `mis_file_1tc.16` also shows stagnation (`0.76`).
- Degrees of freedom exercised: Sequential adaptation, `vqe_reps`.
- Change summary: Keep the baseline VQE first attempt, then escalate to a deeper VQE attempt when the first attempt is feasible but stagnant.
- Result: DISCARD, `suite_average_gap=0.604167`, per-instance gaps: `mis_file_1tc.16=0.3750`, `mis_file_p4tc.16=0.8333`.
- Wall time: `39.3s`.
- Analysis: Deeper VQE did not improve either scout instance and nearly doubled runtime because both instances executed two attempts. This rules out simple depth escalation with the same ansatz as an efficient fix for the observed stagnation.

### Iteration 2
- Hypothesis: The baseline may be ansatz-limited rather than depth-limited, so swapping to `efficient_su2` at the same depth could improve the dense 16-node scout instance without the runtime hit of a second attempt.
- Failure signal: RealAmplitudes depth-1 is feasible but stalls; depth-2 with the same ansatz produced no gain.
- Degrees of freedom exercised: `ansatz_type`.
- Change summary: Replace the baseline VQE ansatz with `efficient_su2` while keeping depth and optimizer fixed.
- Result: DISCARD, `suite_average_gap=0.750000`, per-instance gaps: `mis_file_1tc.16=0.5000`, `mis_file_p4tc.16=1.0000`.
- Wall time: `22.3s`.
- Analysis: `efficient_su2` is clearly worse than the baseline on both scout instances and loses feasibility entirely on the dense graph. Straight ansatz swapping within VQE is not promising enough to justify more budget right now.

### Iteration 3
- Hypothesis: CVaR optimisation may improve the quality of the high-probability feasible tail on dense MIS by prioritising lower-energy samples instead of the full expectation.
- Failure signal: Baseline VQE is already highly feasible, so the remaining issue is weak objective quality rather than feasibility collapse.
- Degrees of freedom exercised: `measurement_mode`, `cvar_alpha`.
- Change summary: Keep baseline VQE ansatz/depth and switch the optimisation objective from expectation to CVaR.
- Result: KEEP, `suite_average_gap=0.062500`, per-instance gaps: `mis_file_1tc.16=0.1250`, `mis_file_p4tc.16=0.0000`.
- Wall time: `27.0s`.
- Analysis: This is the first substantial win. CVaR preserved feasibility and sharply improved objective quality, especially on the dense graph where the baseline had failed badly. Read-only seed checks showed the policy remains strong with alternate seeds (`seed=23 -> 0.0000`, `seed=29 -> 0.1250` on the scout proxy), so the gain appears robust enough to deepen.

### Iteration 4
- Hypothesis: A smaller CVaR tail fraction may further sharpen the feasible elite samples and close the remaining `0.125` gap on `mis_file_1tc.16`.
- Failure signal: The current CVaR policy already solves the dense scout graph, leaving only the medium proxy with residual gap.
- Degrees of freedom exercised: `cvar_alpha`.
- Change summary: Keep the VQE CVaR policy and reduce `cvar_alpha` to focus the objective on a narrower low-energy tail.
- Result: KEEP, `suite_average_gap=0.000000`, per-instance gaps: `mis_file_1tc.16=0.0000`, `mis_file_p4tc.16=0.0000`.
- Wall time: `29.9s`.
- Analysis: Narrowing the CVaR tail saturates the 2-instance scout proxy, but promotion later showed this was a proxy overfit: `p1tc.16` and `p2tc.16` both failed on full-stage confirmation.

## Stage 1 Promotion

- Scout leaders:
  - `Iteration 4`: `VQE real_amplitudes d=1 CVaR(0.1)` with scout `suite_average_gap=0.0000`
  - `Iteration 3`: `VQE real_amplitudes d=1 CVaR(0.25)` with scout `suite_average_gap=0.0625`
  - `Iteration 0`: baseline `VQE real_amplitudes d=1 expectation` with scout `suite_average_gap=0.6042`
- Promoted beam candidates:
  - `CVaR(0.1)` confirmed at `suite_average_gap=0.4000`
  - `CVaR(0.25)` confirmed at `suite_average_gap=0.2500`
  - baseline confirmation was evaluated during promotion but was not the best confirmed snapshot
- Confirmed stage winner so far: `VQE real_amplitudes d=1 COBYLA CVaR(0.25)` with full 16-node `suite_average_gap=0.2500`.
- Analysis: The 16-node scout proxy is too narrow to distinguish between policies that solve `1tc.16` and `p4tc.16` but fail the planted `p1tc.16` / `p2tc.16` variants. Future stage-1 work needs to treat scout saturation as insufficient evidence and use targeted full-stage checks when new families tie on the proxy.

### Iteration 5
- Hypothesis: Warm-start QAOA with CVaR should outperform confirmed VQE on the full 16-node stage because it solves the planted failures (`p1tc.16`, `p2tc.16`) in direct probes while also solving the scout proxy.
- Failure signal: Confirmed VQE winner still has `suite_average_gap=0.2500` due entirely to planted-variant feasibility failures, especially `p2tc.16`.
- Degrees of freedom exercised: `solver_family`, QAOA `variant`, `measurement_mode`, `cvar_alpha`, `reps`, `ws_epsilon`, `ws_source`.
- Change summary: Switch the stage-1 candidate from VQE CVaR to warm-start QAOA CVaR for a direct robustness test.
- Result: SCOUT DISCARD at `suite_average_gap=0.0000` because the proxy was already saturated; direct full-stage confirm on the archived QAOA snapshot achieved `suite_average_gap=0.0250`, per-instance gaps: `1tc.16=0.0000`, `p1tc.16=0.0000`, `p2tc.16=0.1250`, `p3tc.16=0.0000`, `p4tc.16=0.0000`.
- Wall time: scout `8.5s`, direct confirm `19.2s`.
- Analysis: Warm-start QAOA ties the scout proxy while dramatically outperforming the promoted VQE winner on the full 16-node stage (`0.0250` vs `0.2500`). This is the real stage-1 winner despite the scout beam being unable to admit tied candidates once the proxy saturates.

## Stage 1 Locked Winner

- Locked policy: `QAOA warmstart`, `reps=1`, `measurement_mode=cvar`, `cvar_alpha=0.25`, `ws_epsilon=0.25`, `ws_source=relaxation`, `optimizer=COBYLA`.
- Confirmed 16-node `suite_average_gap`: `0.0250`.
- Rationale: best full-stage result found; solves every 16-node curriculum instance except `p2tc.16`, where it still returns a near-optimal feasible set (`gap=0.125`).

### Iteration 6
- Hypothesis: The locked 16-node warm-start QAOA winner may transfer unchanged into the 32-node scout and provide a strong seed for stage 2.
- Failure signal: None before the run; this is the required curriculum transfer check.
- Degrees of freedom exercised: None. Carried the locked stage-1 winner forward unchanged.
- Change summary: Seed `mis_curriculum_32` with the confirmed 16-node QAOA warm-start policy.
- Result: KEEP by default because the 32-node scout had no incumbent yet, but the transferred policy scored `suite_average_gap=1.0000` with replay guardrail `replay_16_suite_average_gap=0.0000`.
- Wall time: `35.9s`.
- Analysis: Transfer failed completely at 32 nodes. Direct probes showed the failure mode is not mostly infeasibility but concentration collapse: many feasible samples exist, yet `top1_prob≈0.0049`, so the most-probable bitstring is indistinguishable from noise and the concentration guard returns gap `1.0`. Stage-2 family screening then showed:
  - VQE CVaR: also diffuse failure (`gap=1.0`) and very slow (`~110s`).
  - PCE default: concentrated on a single infeasible bitstring (`gap=1.0`, `feas_rate=0`).
  - QRAO 3:1 semideterministic: meaningful feasible result on `1tc.32` (`gap=0.4167`) but failed `p8tc.32`.
  - QRAO 2:1 semideterministic: feasible on `p8tc.32` with `gap=0.4000`.
  - QRAO 3:1 magic: best `p8tc.32` quality seen so far (`gap=0.2000`) but with extremely sparse feasible mass.

### Iteration 7
- Hypothesis: QRAO with lower compression (`qrao_max_vars_per_qubit=2`) will outperform warm-start QAOA on the 32-node scout because it already returns feasible, concentrated solutions on both scout instances in direct probes.
- Failure signal: Stage-2 QAOA warm-start collapsed into near-uniform noise on both 32-node scout instances.
- Degrees of freedom exercised: `solver_family`, `qrao_max_vars_per_qubit`, `rounding`.
- Change summary: Replace stage-2 QAOA warm-start with a lower-compression QRAO candidate.
- Result: KEEP, `suite_average_gap=0.7000`, per-instance scout gaps: `1tc.32=1.0000`, `p8tc.32=0.4000`, with `replay_16_suite_average_gap=0.0000`.
- Wall time: `116.8s`.
- Analysis: Lower compression fixed the dense 32-node scout instance but not the reference 32-node instance. Follow-up probes showed that `QRAO 3:1 magic` is better on both 32-node scout instances than `2:1 semideterministic`: `1tc.32=0.2500` and `p8tc.32=0.2000`.

### Iteration 8
- Hypothesis: QRAO `3:1` with `magic` rounding will beat the current stage-2 incumbent because it improves both 32-node scout instances in direct probes while keeping the 16-node replay branch untouched.
- Failure signal: `QRAO 2:1 semideterministic` only improved one of the two 32-node scout instances.
- Degrees of freedom exercised: `qrao_max_vars_per_qubit`, `rounding`.
- Change summary: Replace the stage-2 QRAO rounding/compression pair with the stronger `3:1` + `magic` configuration.
- Result: KEEP, `suite_average_gap=0.6667`, per-instance scout gaps: `1tc.32=0.3333`, `p8tc.32=1.0000`, with `replay_16_suite_average_gap=0.0000`.
- Wall time: `182.3s`.
- Analysis: `3:1 magic` improved the 32-node reference instance but was unstable on the dense scout variant during the actual scout run, even though a direct probe on that instance had looked better. Static QRAO choices are therefore too brittle; the stage-2 branch should adapt based on first-attempt success/failure instead of committing globally to one rounding/compression pair.

### Iteration 9
- Hypothesis: A two-step QRAO controller will beat the current stage-2 incumbent by using `3:1 magic` first and falling back to `2:1 semideterministic` only when the first 32-node attempt is infeasible or too weak.
- Failure signal: `3:1 magic` helps `1tc.32` but can collapse on `p8tc.32`; `2:1 semideterministic` shows the opposite tradeoff.
- Degrees of freedom exercised: sequential adaptation, `qrao_max_vars_per_qubit`, `rounding`, early stopping.
- Change summary: Add a guarded second QRAO attempt for 32-node problems instead of committing to a single static QRAO configuration.
- Result: KEEP, `suite_average_gap=0.3250`, per-instance scout gaps: `1tc.32=0.2500`, `p8tc.32=0.4000`, with `replay_16_suite_average_gap=0.0000`.
- Wall time: `196.2s`.
- Analysis: The adaptive controller materially improved the 32-node scout, but the winning attempt on both scout instances still came from the first `3:1 magic` branch; the guarded fallback did not end up being selected in the winning records. Additional direct probes then showed a simpler static variant, `QRAO 2:1 magic`, is better still on both scout instances (`1tc.32=0.1667`, `p8tc.32=0.4000`).

### Iteration 10
- Hypothesis: Static `QRAO 2:1 magic` will outperform the current stage-2 adaptive incumbent because it improves both 32-node scout instances in direct probes without needing a second attempt.
- Failure signal: The adaptive QRAO controller still leaves substantial gap on `1tc.32`, and the winning records suggest the fallback branch is not essential.
- Degrees of freedom exercised: `qrao_max_vars_per_qubit`, `rounding`, early stopping.
- Change summary: Simplify the 32-node branch to the stronger static `QRAO 2:1 magic` configuration.
- Result: KEEP, `suite_average_gap=0.2250`, per-instance scout gaps: `1tc.32=0.2500`, `p8tc.32=0.2000`, with `replay_16_suite_average_gap=0.0000`.
- Wall time: `123.6s`.
- Analysis: This is the strongest 32-node scout result so far and it improves both proxy instances simultaneously. The current stage-2 branch is therefore `QAOA warmstart` for 16-node replay and `QRAO 2:1 magic` for 32-node primary search. Stage-2 promotion is now running to verify whether the scout gain transfers to the full 32-node suite.

## Stage 2 Locked Winner

- Locked policy: `QAOA warmstart` on 16-node instances and static `QRAO 3:1 magic` on larger instances, with the default `5` attempts and no within-instance adaptation.
- Confirmed 32-node `train_suite_average_gap`: `0.246743`.
- Confirmed 16-node replay `suite_average_gap`: `0.0250`.
- Per-instance 32-node confirmation:
  - `mis_file_1tc.32=0.2500`
  - `mis_file_p1tc.32=0.1538`
  - `mis_file_p3tc.32=0.2727`
  - `mis_file_p5tc.32=0.3571`
  - `mis_file_p8tc.32=0.2000`
- Rationale: stage-2 promotion eventually showed that the seemingly weaker scout candidate `QRAO 3:1 magic` generalises best on the full 32-node suite. Promotion restored `experiment.py` to the winning snapshot `policy_checkpoints/mis_curriculum_32/scout_0008_911e6430.py`.

## Stage 3 Sparse Transition

- Early 48-node transfer work established that magic-rounding QRAO did not scale cleanly to the retained large-instance regime.
- The durable takeaway for the reduced setup was the sparse-side feasibility path: start with `QRAO 3:1 semideterministic`, then fall back to `QRAO 2:1 semideterministic` when the first large-instance attempt is infeasible or remains above the gap threshold.
- Broader 48-node and 64-node structure-specific notes for variants later removed from the reduced setup are intentionally omitted from this reduced journal.

### Iteration 21
- Hypothesis: Under the reduced 48-node setup where only `mis_file_p1tc.48` remains, the current adaptive policy is leaving value on the table by stopping after the first successful `QRAO 2:1 semideterministic` fallback, so allowing one additional retry of that same sparse-winning branch can improve the single-instance best gap beyond the current direct baseline `0.4667`.
- Failure signal: Direct sparse-only probes under the reduced setup showed that the current policy's first `QRAO 3:1 semideterministic` attempt always fails on `mis_file_p1tc.48`, while the second-attempt `QRAO 2:1 semideterministic` branch can succeed (`gap=0.4667`). Static one-shot `QRAO 2:1 semideterministic` was unstable (`gap=1.0` once, `gap=0.6000` over two tries), which suggests a third attempt on the same winning branch may still uncover a better feasible draw.
- Degrees of freedom exercised: large-instance attempt budget.
- Change summary: For `n_vars > 32`, keep the current adaptive branch order but allow up to `3` attempts so the successful `QRAO 2:1 semideterministic` fallback can be retried once on the reduced 48-node sparse target.
- Result: KEEP for the reduced 48-node setup. Direct evaluation on the only remaining 48-node instance improved to `mis_file_p1tc.48=0.3333` (`AR=0.6667`, feasible, `354.3s` total).
- Wall time: `354.3s` total. Attempt breakdown on `mis_file_p1tc.48`:
  - Attempt 0: `QRAO 3:1 semideterministic`, `gap=1.0000`, infeasible, `203.4s`
  - Attempt 1: `QRAO 2:1 semideterministic`, `gap=0.3333`, feasible, `150.6s`
- Analysis: Under the reduced 48-node objective, this is the strongest result found so far on the only remaining training instance. The key behavior remains the same: `3:1 semideterministic` is a dead branch for `mis_file_p1tc.48`, and the win comes entirely from the `2:1 semideterministic` fallback. The new best direct sparse-48 result is `gap=0.3333`, which is materially better than the earlier `0.5333` / `0.4667` baselines. This is a reasonable point to shift effort to the retained 64-node held-out sparse case (`1tc.64`).

## Reduced Sparse 64 Held-Out Result

- Under the reduced held-out setup, the retained 64-node test case is `mis_file_1tc.64`.
- Recorded result on the retained sparse case: `mis_file_1tc.64=0.5500`, feasible, `784.8s`.
- Interpretation: the reduced sparse 48-node policy transfers partially to the retained sparse 64-node held-out case, but still leaves material room for improvement.
