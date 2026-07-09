# CVRP Agent Journal

This journal tracks the AutoQResearch search for CVRP policies. It is separate from the MIS journal and should stay under `cvrp_results/`.

## Baseline

Required baseline:

- GAP solver: VQE
- Route solver: classical exact TSP
- Ansatz: `efficient_su2`
- VQE depth: `vqe_reps=1`
- Measurement: expectation
- Seed heuristic: `depot_farthest`
- GAP capacity penalty: `hard_slack`
- Local search: disabled
- Seed: `17`

## Instance Ladder

- Stage 1: `cvrp_curriculum_8`, backed by `individual/cvrp/Synth-n9-k2-s0.vrp`, `Synth-n9-k2-s1.vrp`, and `Synth-n9-k2-s2.vrp`
- Stage 2: `cvrp_curriculum_10`, backed by `individual/cvrp/Synth-n11-k2-s0.vrp` and `Synth-n11-k2-s1.vrp`
- Stage 3: `cvrp_curriculum_12`, backed by `individual/cvrp/Synth-n13-k3-s0.vrp`
- Held-out final: `cvrp_benchmark_e13`, backed by `individual/cvrp/E-n13-k4.vrp`

## Initial Roadmap

1. Run the baseline scout on `cvrp_curriculum_8` and record the stage-1 starting gap.
2. Diagnose whether the main failure is GAP feasibility, poor concentration, or poor routed cost after a feasible assignment.
3. Compare GAP-stage solver families first while keeping routes classical: VQE, CVaR VQE, QAOA, CVaR QAOA, PCE, and QRAO.
4. Once GAP assignments are reliably feasible, test route-stage quantum solvers on clusters that fit the route qubit threshold.
5. Only after a stable stage-1 policy emerges, promote it and move to `cvrp_curriculum_10`, then `cvrp_curriculum_12`.
6. Run `cvrp_benchmark_e13` only after the policy is locked.

## Iteration 0 - Baseline Scout

- Hypothesis: The conservative baseline gives a reproducible reference for hard-slack Fisher-Jaikumar GAP clustering with classical route scoring.
- Failure signal: GAP feasibility is weak: `cvrp_8_s1` produced no feasible decoded assignment, and `cvrp_8_s0` had only `raw_feasibility_rate=0.001`.
- Degrees of freedom exercised: none; this is the required baseline.
- Change summary: none.
- Command:

```bash
./.venv/bin/python agent_harness.py --single-run --suite cvrp_curriculum_8 --eval-workflow scout --no-dev --branch codex/cvrp01
```

- Result: KEEP. `suite_average_gap=0.608209`.
- Per-instance gaps: `cvrp_8_s0=0.216418`; `cvrp_8_s1=1.000000`.
- Wall time: harness wall time `139.1s`; suite total wall time `138.6s`.
- Feasibility notes: `cvrp_8_s0` feasible with `raw_ar=0.7836` and `raw_feasibility_rate=0.001`; `cvrp_8_s1` infeasible with `raw_ar=0.0` and `raw_feasibility_rate=0.0`.
- Next experiment: keep hard-slack GAP and classical route scoring, but switch the VQE GAP objective from expectation to CVaR to see whether the low-cost sampled tail contains more feasible assignments.

## Iteration 1 - CVaR VQE GAP

- Hypothesis: The baseline expectation-value VQE spreads probability across mostly infeasible GAP assignments; CVaR VQE should bias optimization toward low-energy sampled assignments and may raise the chance of decoding a feasible cluster on `cvrp_8_s1`.
- Failure signal: Baseline `cvrp_8_s1` scored `gap=1.000000` with `raw_feasible=false`, while `cvrp_8_s0` had only a single-shot-level feasibility rate.
- Degrees of freedom exercised: GAP-stage VQE measurement mode only: `measurement_mode="cvar"` with `cvar_alpha=0.25`. Route stage remains `classical`; `cvrp_seed_method="depot_farthest"` and `cvrp_gap_penalty_method="hard_slack"` remain unchanged.
- Change summary: set the CVRP VQE base policy to CVaR measurement; keep `seed=17`, `pce_local_search=False`, and `final_local_search=False`.
- Result: DISCARD. The harness was manually stopped after more than 10 minutes with no completed suite record; it was already far slower than the `139.1s` baseline scout.
- Per-instance gaps: no completed suite summary was written before interruption.
- Wall time: greater than 10 minutes before manual stop.
- Feasibility notes: no saved partial result was available; the long sampler-based optimizer cost is the primary signal.
- Next experiment: switch the GAP solver family to warm-start QAOA with expectation measurement, preserving classical routes and the same CVRP seed and hard-slack penalty settings.

## Iteration 2 - Warm-Start QAOA GAP

- Hypothesis: A depth-1 warm-start QAOA circuit should provide a cheaper structured search over the GAP QUBO than CVaR VQE, while the relaxation warm start may improve assignment feasibility on `cvrp_8_s1`.
- Failure signal: Baseline VQE expectation had near-zero feasible mass, and CVaR VQE was too slow for scout use.
- Degrees of freedom exercised: GAP solver family and QAOA variant: `gap_solver_family="qaoa"`, `variant="warmstart"`, `reps=1`, `measurement_mode="expectation"`, `ws_source="relaxation"`, `ws_epsilon=0.25`.
- Change summary: make CVRP start with QAOA for the GAP stage; keep `route_solver_family="classical"`, `cvrp_seed_method="depot_farthest"`, `cvrp_gap_penalty_method="hard_slack"`, `seed=17`, and local search disabled.
- Result: DISCARD. The harness was manually stopped after the child process was still on `cvrp_8_s0` beyond the first scout instance's practical budget.
- Per-instance gaps: no completed suite summary was written before interruption.
- Wall time: greater than 6 minutes before manual stop.
- Feasibility notes: no saved partial result was available; the QAOA Pauli-evolution circuit was too slow for this scout loop before feasibility could be measured.
- Next experiment: test QRAO compression on the GAP QUBO with expectation measurement, using 2 variables per qubit to reduce qubit count without the more aggressive 3-to-1 compression.

## Iteration 3 - QRAO 2-to-1 GAP

- Hypothesis: QRAO should reduce the 26-qubit hard-slack GAP problem to a smaller variational circuit, and 2-to-1 encoding may preserve more assignment structure than 3-to-1 while still being much cheaper than QAOA.
- Failure signal: VQE expectation produced near-zero feasible mass; CVaR VQE and warm-start QAOA were too slow to serve as scout policies.
- Degrees of freedom exercised: GAP solver family and QRAO encoding: `gap_solver_family="qrao"`, `qrao_max_vars_per_qubit=2`, `qrac_type=2`, `rounding="semideterministic"`, `ansatz_type="real_amplitudes"`, `vqe_reps=1`, `measurement_mode="expectation"`.
- Change summary: make CVRP start with QRAO for the GAP stage and restore the QAOA branch to its prior non-CVRP behavior; keep classical routes, hard-slack penalty, depot-farthest seeds, `seed=17`, and local search disabled.
- Result: DISCARD. `suite_average_gap=1.000000`.
- Per-instance gaps: `cvrp_8_s0=1.000000`; `cvrp_8_s1=1.000000`.
- Wall time: harness wall time `431.0s`; suite total wall time `430.4s`.
- Feasibility notes: both scout instances were infeasible with `raw_feasibility_rate=0.0`; QRAO used 13 qubits, depth 14, and 5 attempts per instance, but did not recover any feasible decoded GAP assignment.
- Next experiment: return to baseline VQE expectation and change the Fisher-Jaikumar seed heuristic from depot-farthest to angle-spread.

## Iteration 4 - Angle-Spread VQE GAP

- Hypothesis: Depot-farthest seeds may be over-focusing on radial extremes; angle-spread seeds should better separate customer sectors and may make the hard-slack GAP assignment easier for VQE to sample feasibly.
- Failure signal: The solver-family changes either timed out or destroyed feasibility, while the original VQE policy was at least feasible on `cvrp_8_s0`.
- Degrees of freedom exercised: CVRP seed heuristic only: `cvrp_seed_method="angle_spread"`. GAP solver stays VQE expectation with EfficientSU2 depth 1; route solver stays classical.
- Change summary: set the CVRP base policy seed method to angle-spread; keep hard-slack capacity penalties, `seed=17`, and local search disabled.
- Result: DISCARD. `suite_average_gap=0.608209`, equal to the incumbent and therefore not a strict improvement.
- Per-instance gaps: `cvrp_8_s0=0.216418`; `cvrp_8_s1=1.000000`.
- Wall time: harness wall time `566.4s`; suite total wall time `565.9s`.
- Feasibility notes: `cvrp_8_s0` remained feasible with `raw_feasibility_rate=0.001`; `cvrp_8_s1` remained infeasible with `raw_feasibility_rate=0.0`. The adaptive candidate burned 5 repeated VQE attempts per instance without changing the selected solution.
- Next experiment: keep depot-farthest seeds and VQE expectation, but replace hard-slack capacity encoding with the taylor penalty to reduce the scout GAP from 26 to 16 QUBO variables.

## Iteration 5 - Taylor Penalty VQE GAP

- Hypothesis: The taylor capacity penalty removes slack variables, dropping the scout GAP from 26 to 16 QUBO variables. The smaller circuit should be faster and may concentrate more feasible assignment samples while preserving the depot-farthest seed geometry that has an exact seed-heuristic optimum on both scout instances.
- Failure signal: Hard-slack VQE has almost no feasible sampled mass, and repeated attempts do not help; QRAO compression also produced no feasible samples.
- Degrees of freedom exercised: CVRP GAP capacity encoding only: `cvrp_gap_penalty_method="taylor"` with `cvrp_taylor_alpha=10.0`. Solver remains VQE expectation; routes remain classical; seeds remain depot-farthest.
- Change summary: change the CVRP base policy capacity penalty from hard-slack to taylor; keep `seed=17`, `pce_local_search=False`, and `final_local_search=False`.
- Result: DISCARD/INVALID. The harness was stopped before completion after discovering that non-baseline adaptive runs construct CVRP problems before `build_base_policy`, so changing `cvrp_gap_penalty_method` in the base policy did not actually rebuild the GAP QUBO.
- Per-instance gaps: no completed suite summary was written before interruption.
- Wall time: stopped during `cvrp_8_s0`; no metric-bearing result.
- Feasibility notes: this run was still operating on the default hard-slack problem, so it cannot answer the taylor-penalty hypothesis.
- Next experiment: use only solver-side knobs that apply after problem construction; also add CVRP one-attempt stopping because five deterministic repeated VQE attempts reproduced the same result at 4x wall time.

## Iteration 6 - Real-Amplitudes VQE Single Attempt

- Hypothesis: `real_amplitudes` halves the CVRP VQE parameter count relative to `efficient_su2` and may improve optimizer behavior or runtime without changing GAP construction. CVRP one-attempt stopping avoids repeating deterministic policies that have shown no benefit.
- Failure signal: `efficient_su2` hard-slack VQE finds a feasible solution on `cvrp_8_s0` but not `cvrp_8_s1`; five repeated attempts did not improve the angle-spread probe and only increased wall time.
- Degrees of freedom exercised: solver-side VQE ansatz and continuation policy: CVRP `ansatz_type="real_amplitudes"`, `vqe_reps=1`, `measurement_mode="expectation"`, and stop CVRP after the first attempt.
- Change summary: restore hard-slack CVRP penalty, change the CVRP VQE ansatz to real-amplitudes, and add CVRP early stopping after one attempt; keep classical routes, depot-farthest seeds, `seed=17`, and local search disabled.
- Result: DISCARD. `suite_average_gap=1.000000`.
- Per-instance gaps: `cvrp_8_s0=1.000000`; `cvrp_8_s1=1.000000`.
- Wall time: harness wall time about `107.0s`; suite total wall time `107.0s`.
- Feasibility notes: both instances were infeasible with `raw_feasibility_rate=0.0`; the 52-parameter real-amplitudes circuit was faster than EfficientSU2 but lost the one feasible `cvrp_8_s0` decode.
- Next experiment: restore EfficientSU2, keep one-attempt CVRP stopping, and increase final `sampler_shots` to improve the chance of observing rare feasible assignments without changing optimizer shots.

## Iteration 7 - EfficientSU2 With More Final Samples

- Hypothesis: Baseline EfficientSU2 has a tiny but nonzero feasible probability on `cvrp_8_s0`; increasing final sampler shots from 1024 to 4096 may expose rare feasible assignments on `cvrp_8_s1` while preserving the baseline optimizer trajectory.
- Failure signal: Baseline `cvrp_8_s0` feasibility rate was `0.001`, and `cvrp_8_s1` had no feasible samples at 1024 shots.
- Degrees of freedom exercised: CVRP final sampling and continuation policy: `sampler_shots=4096` for CVRP and one-attempt stopping. VQE remains EfficientSU2 depth 1 with expectation measurement and COBYLA 150 iterations.
- Change summary: override CVRP `sampler_shots` to 4096 and stop CVRP after one attempt; keep hard-slack penalties, depot-farthest seeds, `seed=17`, `pce_local_search=False`, and `final_local_search=False`.
- Result: KEEP. `suite_average_gap=0.172828`.
- Per-instance gaps: `cvrp_8_s0=0.166667`; `cvrp_8_s1=0.178988`.
- Wall time: harness wall time `194.0s`; suite total wall time `193.4s`.
- Feasibility notes: both scout instances became feasible. `cvrp_8_s0` had `raw_feasibility_rate=0.001`; `cvrp_8_s1` had `raw_feasibility_rate=0.0002`, indicating the keep came from very rare feasible samples rather than broad concentration.
- Next experiment: increase CVRP final sampler shots to 8192 to see whether additional rare feasible samples improve routed cost further.

## Iteration 8 - EfficientSU2 With 8192 Final Samples

- Hypothesis: If 4096 shots finds only one or a few feasible assignments, 8192 final samples may recover lower routed-cost feasible decodes, especially on `cvrp_8_s1`, without changing the optimizer trajectory.
- Failure signal: The kept 4096-shot policy is feasible but still has nontrivial routed gaps around `0.17` and extremely low feasibility rates.
- Degrees of freedom exercised: CVRP final sampling only: raise `sampler_shots` from 4096 to 8192. Keep one-attempt stopping and the same EfficientSU2 expectation VQE.
- Change summary: update CVRP `sampler_shots` to 8192; keep hard-slack penalties, depot-farthest seeds, `seed=17`, and local search disabled.
- Result: KEEP. `suite_average_gap=0.112352`.
- Per-instance gaps: `cvrp_8_s0=0.166667`; `cvrp_8_s1=0.058036`.
- Wall time: harness wall time `284.4s`; suite total wall time `283.8s`.
- Feasibility notes: both instances remained feasible. `cvrp_8_s0` had `raw_feasibility_rate=0.0005`; `cvrp_8_s1` had `raw_feasibility_rate=0.0004`. More shots improved `cvrp_8_s1` substantially but did not improve `cvrp_8_s0`.
- Next experiment: increase CVRP final sampler shots to 16384 to test whether the rare-feasible-sample effect continues before changing optimizer or ansatz depth.

## Iteration 9 - EfficientSU2 With 16384 Final Samples

- Hypothesis: Feasible samples remain rare at 8192 shots, so 16384 final samples may find a lower-cost feasible decode, especially for `cvrp_8_s0` where the current gap is stuck at `0.166667`.
- Failure signal: The kept 8192-shot policy still has tiny feasibility rates and nonzero gaps on both scout instances.
- Degrees of freedom exercised: CVRP final sampling only: raise `sampler_shots` from 8192 to 16384. Keep one-attempt stopping and the same EfficientSU2 expectation VQE.
- Change summary: update CVRP `sampler_shots` to 16384; keep hard-slack penalties, depot-farthest seeds, `seed=17`, `pce_local_search=False`, and `final_local_search=False`.
- Result: DISCARD. `suite_average_gap=0.112352`, equal to the incumbent and slower.
- Per-instance gaps: `cvrp_8_s0=0.166667`; `cvrp_8_s1=0.058036`.
- Wall time: suite total wall time `471.0s`.
- Feasibility notes: both instances remained feasible, but feasible rates stayed tiny (`0.0004` and `0.0005`) and no better routed assignment was found.
- Next experiment: return to 8192 final samples and test circular EfficientSU2 entanglement as a modest circuit-structure change.

## Iteration 10 - Circular EfficientSU2

- Hypothesis: Circular entanglement may improve probability flow across the GAP assignment variables compared with linear entanglement, possibly improving `cvrp_8_s0` without the large wall-time hit from 16384 samples.
- Failure signal: 8192 final samples improved `cvrp_8_s1` but left `cvrp_8_s0` at `gap=0.166667`; 16384 samples did not improve either instance.
- Degrees of freedom exercised: VQE entanglement only: `entanglement="circular"` with EfficientSU2 depth 1, expectation measurement, one-attempt stopping, and `sampler_shots=8192`.
- Change summary: restore `sampler_shots=8192` and set CVRP VQE entanglement to circular; keep hard-slack penalties, depot-farthest seeds, `seed=17`, and local search disabled.
- Result: DISCARD. `suite_average_gap=0.179119`.
- Per-instance gaps: `cvrp_8_s0=0.166667`; `cvrp_8_s1=0.191571`.
- Wall time: suite total wall time `287.2s`.
- Feasibility notes: both instances stayed feasible, and `cvrp_8_s0` feasibility rate rose to `0.0012`, but routed quality on `cvrp_8_s1` regressed relative to the 8192-shot linear policy.
- Next experiment: restore linear entanglement and increase COBYLA iterations to test whether more optimizer work improves the sampled feasible assignments.

## Iteration 11 - EfficientSU2 COBYLA 300

- Hypothesis: The current kept policy is sample-limited but may also be under-optimized at 150 COBYLA iterations; allowing 300 iterations could improve the final state enough to reduce routed gaps without changing ansatz or sampling.
- Failure signal: `cvrp_8_s0` remains at `gap=0.166667` across 8192 and 16384 shots, suggesting the optimizer state may not put better assignments into the sampled support.
- Degrees of freedom exercised: VQE optimizer budget only: raise `optimizer_maxiter` from 150 to 300 while keeping EfficientSU2 linear, expectation measurement, one-attempt stopping, and `sampler_shots=8192`.
- Change summary: set CVRP optimizer max iterations to 300; keep hard-slack penalties, depot-farthest seeds, `seed=17`, and local search disabled.
- Result: DISCARD. `suite_average_gap=0.597701`.
- Per-instance gaps: `cvrp_8_s0=0.195402`; `cvrp_8_s1=1.000000`.
- Wall time: suite total wall time `351.5s`.
- Feasibility notes: `cvrp_8_s0` stayed feasible but regressed; `cvrp_8_s1` became infeasible with `raw_feasibility_rate=0.0`. More COBYLA iterations moved the optimizer away from the rare useful feasible region.
- Next experiment: keep the 150-iteration budget and test SPSA as a different optimizer path.

## Iteration 12 - EfficientSU2 SPSA

- Hypothesis: SPSA may explore the 104-parameter EfficientSU2 landscape differently from COBYLA and recover feasible low-cost assignments without increasing max iterations.
- Failure signal: COBYLA at 300 iterations regressed badly; the 150-iteration COBYLA incumbent still depends on rare feasible samples.
- Degrees of freedom exercised: VQE optimizer method only: `optimizer_method="SPSA"` with `optimizer_maxiter=150`, EfficientSU2 linear, expectation measurement, one-attempt stopping, and `sampler_shots=8192`.
- Change summary: switch the CVRP optimizer method to SPSA; keep hard-slack penalties, depot-farthest seeds, `seed=17`, and local search disabled.
- Result: DISCARD. `suite_average_gap=1.000000`.
- Per-instance gaps: `cvrp_8_s0=1.000000`; `cvrp_8_s1=1.000000`.
- Wall time: suite total wall time `377.9s`.
- Feasibility notes: both instances were infeasible with `raw_feasibility_rate=0.0`; SPSA also ran 301 optimizer iterations internally, so it was slower and worse than COBYLA.
- Next experiment: restore COBYLA and test the `pauli_two_design` VQE ansatz while keeping the accepted 8192-shot sampling and one-attempt stopping.

## Iteration 13 - Pauli Two Design VQE

- Hypothesis: `pauli_two_design` may explore a different variational state family than EfficientSU2 while keeping depth 1 and could put better feasible assignments into the sampled support.
- Failure signal: EfficientSU2 remains nonzero-gap on both scout instances; real-amplitudes was too weak and circular EfficientSU2 worsened `cvrp_8_s1`.
- Degrees of freedom exercised: VQE ansatz only: `ansatz_type="pauli_two_design"` with `vqe_reps=1`, expectation measurement, COBYLA 150, one-attempt stopping, and `sampler_shots=8192`.
- Change summary: switch the CVRP VQE ansatz to pauli-two-design; keep hard-slack penalties, depot-farthest seeds, `seed=17`, and local search disabled.
- Result: DISCARD. `suite_average_gap=0.584980`.
- Per-instance gaps: `cvrp_8_s0=0.169960`; `cvrp_8_s1=1.000000`.
- Wall time: suite total wall time `225.4s`.
- Feasibility notes: `cvrp_8_s0` was barely feasible with `raw_feasibility_rate=0.0001`, while `cvrp_8_s1` was infeasible. The shallower 52-parameter ansatz did not preserve the useful EfficientSU2 feasible support.
- Next experiment: restore EfficientSU2 and test `vqe_reps=2` for more expressivity at the accepted 8192-shot sampler budget.

## Iteration 14 - EfficientSU2 Depth 2

- Hypothesis: Increasing EfficientSU2 to `vqe_reps=2` may represent a better GAP assignment distribution and reduce the persistent `cvrp_8_s0` gap that sampling alone did not improve.
- Failure signal: Lower-parameter ansatz variants lost feasibility; the current EfficientSU2 depth-1 policy is feasible but still has a nonzero routed gap on both instances.
- Degrees of freedom exercised: VQE ansatz depth only: `vqe_reps=2` with EfficientSU2 linear, expectation measurement, COBYLA 150, one-attempt stopping, and `sampler_shots=8192`.
- Change summary: restore EfficientSU2 and raise CVRP `vqe_reps` to 2; keep hard-slack penalties, depot-farthest seeds, `seed=17`, and local search disabled.
- Result: DISCARD. `suite_average_gap=0.574899`.
- Per-instance gaps: `cvrp_8_s0=0.149798`; `cvrp_8_s1=1.000000`.
- Wall time: suite total wall time `296.7s`.
- Feasibility notes: depth 2 slightly improved `cvrp_8_s0` but made `cvrp_8_s1` infeasible with `raw_feasibility_rate=0.0`; the extra CNOTs and parameters did not generalize across the scout pair.
- Next experiment: restore depth 1 and test a higher estimator-shot budget to reduce optimizer noise while preserving the accepted final sampler budget.

## Iteration 15 - EfficientSU2 Estimator 2048

- Hypothesis: The accepted policy may be sensitive to noisy expectation estimates during COBYLA optimization; raising `estimator_shots` from 1024 to 2048 may improve the optimized state without changing ansatz depth or final sample count.
- Failure signal: Increasing final samples beyond 8192 did not help, and deeper or alternate ansatzes lost feasibility on `cvrp_8_s1`.
- Degrees of freedom exercised: CVRP optimization sampling only: `estimator_shots=2048` with `sampler_shots=8192`, EfficientSU2 depth 1, COBYLA 150, expectation measurement, and one-attempt stopping.
- Change summary: set CVRP `estimator_shots` to 2048; keep hard-slack penalties, depot-farthest seeds, `seed=17`, and local search disabled.
- Result: DISCARD. `suite_average_gap=0.112352`, equal to the incumbent.
- Per-instance gaps: `cvrp_8_s0=0.166667`; `cvrp_8_s1=0.058036`.
- Wall time: suite total wall time `286.6s`.
- Feasibility notes: both instances stayed feasible with the same routed gaps as the 8192-shot incumbent; estimator-shot increase did not improve the decoded solution.
- Next experiment: stop scout probing and run stage-1 beam promotion, restoring the best confirmed policy before moving beyond the 8-customer scout.

## Stage 1 Promotion

- Hypothesis: The best scout policy should remain competitive on the full `cvrp_curriculum_8` suite, including `cvrp_8_s2`, before moving to stage 2.
- Failure signal: A scout-only improvement could overfit the two-instance `cvrp_scout_8` proxy or fail on the third 8-customer curriculum instance.
- Degrees of freedom exercised: none; this is confirmation of beam snapshots.
- Change summary: run beam promotion with `--promote-top-k 3 --restore-best`.
- Result: CONFIRM. Best confirmed `train_suite_average_gap=0.095016` on `cvrp_curriculum_8`.
- Per-instance gaps: `cvrp_8_s0=0.166667`; `cvrp_8_s1=0.058036`; `cvrp_8_s2=0.060345`.
- Wall time: best confirmed snapshot wall time `429.1s`; suite total wall time `428.4s`.
- Feasibility notes: all three stage-1 instances were feasible. The restored best snapshot was `scout_0012_5ffe9468.py`, which keeps the 8192 final sampler policy and adds `estimator_shots=2048`; it tied the 8192-only snapshot on metric and was slightly faster in promotion.
- Next experiment: run the restored policy on `cvrp_curriculum_10` as the stage-2 scout baseline before making further edits.

## Stage 2 - Initial Scout

- Hypothesis: The stage-1 restored policy may transfer to the 10-customer CVRP scout because it improved feasibility through sampling rather than fitting a specific route-stage setting.
- Failure signal: Larger 10-customer GAP QUBOs may make the rare-feasible-sample strategy insufficient or too slow.
- Degrees of freedom exercised: none; evaluate the restored stage-1 policy on `cvrp_curriculum_10`.
- Change summary: no policy edit before this evaluation.
- Result: KEEP as the initial stage-2 incumbent only. `suite_average_gap=1.000000` on `cvrp_curriculum_10`; replay-8 guardrail remained `0.112352`.
- Per-instance gaps: `cvrp_10_s0=1.000000`; `cvrp_10_s1=1.000000`; replay `cvrp_8_s0=0.166667`; replay `cvrp_8_s1=0.058036`.
- Wall time: train suite total wall time `348.5s`; replay-8 total wall time `285.5s`; harness wall time `634.7s`.
- Feasibility notes: both 10-customer instances were infeasible with `raw_feasibility_rate=0.0`; the 8-customer replay remained feasible with the same routed gaps as the stage-1 scout incumbent.
- Next experiment: shift budget toward final GAP sampling on the 10-customer scout, because the failure mode is zero feasible sampled assignments rather than a replay regression.

## Iteration 16 - Stage 2 High Final Sampling

- Hypothesis: The 10-customer GAP state may still place feasible assignments in very low-probability support; increasing final sampler shots can recover at least one feasible decode. To keep wall time bounded, reduce optimizer estimator shots back to the stage-1 kept 1024-shot trajectory while raising final sampling.
- Failure signal: `cvrp_10_s0` and `cvrp_10_s1` both had `raw_feasibility_rate=0.0` at 8192 final samples, while 8-customer replay stayed feasible.
- Degrees of freedom exercised: CVRP sampling allocation only: remove the 2048 estimator-shot override and raise `sampler_shots` from 8192 to 32768. Keep EfficientSU2 depth 1, expectation measurement, COBYLA 150, one-attempt stopping, classical route solving, `seed=17`, and local search disabled.
- Change summary: set CVRP `sampler_shots=32768` and return `estimator_shots` to the default 1024 for CVRP.
- Result: DISCARD. `suite_average_gap=1.000000` on `cvrp_curriculum_10`; replay-8 regressed to `0.583333`.
- Per-instance gaps: `cvrp_10_s0=1.000000`; `cvrp_10_s1=1.000000`; replay `cvrp_8_s0=0.166667`; replay `cvrp_8_s1=1.000000` due timeout.
- Wall time: train suite total wall time `1021.2s`; replay-8 total wall time `801.2s`; harness wall time `1823.3s`.
- Feasibility notes: both 10-customer instances still had `raw_feasibility_rate=0.0`, so 32768 final samples did not rescue feasibility. Replay `cvrp_8_s0` stayed feasible but slowed to `381.3s`; replay `cvrp_8_s1` timed out after `420.0s`, making the candidate unacceptable even aside from unchanged train gap.
- Next experiment: stop spending final-sampling budget and test a CVRP construction knob correctly, because the larger GAP problem appears to need a different clustering QUBO formulation rather than more samples from the current one.

## Iteration 17 - Tilted Capacity Penalty

- Hypothesis: The hard-slack GAP construction adds capacity slack variables and produces a 30-qubit 10-customer QUBO with zero feasible samples. The tilted capacity penalty removes explicit capacity slacks and may produce a lower-dimensional, smoother GAP objective that VQE can sample feasible cluster assignments from.
- Failure signal: The current hard-slack policy stayed infeasible on both 10-customer instances even at 32768 final samples; the failure is not solved by more sampling.
- Degrees of freedom exercised: CVRP GAP construction only: switch `cvrp_gap_penalty_method` from `hard_slack` to `tilted` while keeping depot-farthest seeds, EfficientSU2 depth 1, expectation measurement, COBYLA 150, `estimator_shots=2048`, `sampler_shots=8192`, classical route solving, `seed=17`, and local search disabled.
- Change summary: set the adaptive CVRP construction default and the CVRP policy field to `tilted`; leave the fixed static baseline helper unchanged.
- Result: DISCARD by replay guardrail. `suite_average_gap=0.047210` on `cvrp_curriculum_10`; replay-8 regressed to `0.134564` versus incumbent `0.112352`.
- Per-instance gaps: `cvrp_10_s0=0.033195`; `cvrp_10_s1=0.061224`; replay `cvrp_8_s0=0.086957`; replay `cvrp_8_s1=0.182171`.
- Wall time: train suite total wall time `117.2s`; replay-8 total wall time `41.2s`; harness wall time `159.3s`.
- Feasibility notes: tilted capacity reduced the 10-customer GAP to 20 qubits, made both train instances feasible, and raised raw feasibility rates to `0.1093` and `0.1757`. Replay-8 also stayed feasible with high raw feasibility rates (`0.4274` and `0.2094`), but `cvrp_8_s1` routed quality regressed enough to fail the guardrail by about `0.0022` average gap.
- Next experiment: keep tilted capacity and modestly raise final samples to 16384, because the replay failure is a small decode-quality miss with broad feasible support rather than a feasibility collapse.

## Iteration 18 - Tilted Capacity With 16384 Final Samples

- Hypothesis: The tilted GAP formulation has high feasible-sample rates on replay and train; doubling final samples from 8192 to 16384 may recover a slightly better replay `cvrp_8_s1` routed assignment while preserving the strong 10-customer improvement.
- Failure signal: Iteration 17 missed the replay guardrail by a small margin: replay average `0.134564` versus an allowed bound near `0.132352`.
- Degrees of freedom exercised: final sampling on the tilted CVRP construction only: keep tilted capacity, depot-farthest seeds, EfficientSU2 depth 1, expectation measurement, COBYLA 150, `estimator_shots=2048`, classical route solving, `seed=17`, and local search disabled; raise `sampler_shots` from 8192 to 16384.
- Change summary: reapply tilted adaptive CVRP construction and set CVRP `sampler_shots=16384`.
- Result: KEEP. `suite_average_gap=0.047210` on `cvrp_curriculum_10`; replay-8 improved to `0.091085`.
- Per-instance gaps: `cvrp_10_s0=0.033195`; `cvrp_10_s1=0.061224`; replay `cvrp_8_s0=0.000000`; replay `cvrp_8_s1=0.182171`.
- Wall time: train suite total wall time `152.8s`; replay-8 total wall time `48.2s`; harness wall time `201.9s`.
- Feasibility notes: both 10-customer instances were feasible with raw feasibility rates `0.1116` and `0.1744`; replay-8 was also fully feasible and passed guardrail despite `cvrp_8_s1` remaining worse than the hard-slack stage-1 route.
- Next experiment: run stage-2 beam promotion and restore the best confirmed `cvrp_curriculum_10` policy before considering stage 3.

## Stage 2 Promotion

- Hypothesis: The tilted 16384-sample snapshot should remain the best stage-2 candidate when confirmed against the fuller promotion replay set, because it was the only candidate to make both 10-customer train instances feasible while passing replay guardrails.
- Failure signal: A scout keep could overfit the two-instance 10-customer scout or lose the third 8-customer replay instance.
- Degrees of freedom exercised: none; this is confirmation of beam snapshots.
- Change summary: run stage-2 beam promotion with `--promote-top-k 3 --restore-best`.
- Result: CONFIRM. Best confirmed snapshot `scout_0016_06f9787d.py` with `train_suite_average_gap=0.047210` and `replay_8_suite_average_gap=0.095017`.
- Per-instance gaps: train `cvrp_10_s0=0.033195`; train `cvrp_10_s1=0.061224`; replay `cvrp_8_s0=0.000000`; replay `cvrp_8_s1=0.182171`; replay `cvrp_8_s2=0.102881`.
- Wall time: best confirmed snapshot wall time `230.0s`; train suite wall time `154.1s`; replay-8 wall time `74.9s`.
- Feasibility notes: all train and replay instances were feasible. The 8192-sample tilted snapshot tied the 10-customer train metric but had worse full replay (`0.124003`), and the hard-slack stage-2 incumbent stayed infeasible on 10-customer train (`1.000000`).
- Next experiment: start stage-3 scout on `cvrp_curriculum_12` with the restored tilted 16384-sample policy.

## Stage 3 - Initial Scout

- Hypothesis: The promoted tilted 16384-sample policy may transfer to the 12-customer CVRP target because it solved the 10-customer train pair with high feasible-sample rates and reduced the GAP qubit count.
- Failure signal: Scaling from 10 to 12 customers increases the tilted GAP QUBO to 36 variables and may again make feasible assignments too rare for the current seed geometry.
- Degrees of freedom exercised: none; evaluate the restored stage-2 policy on `cvrp_curriculum_12`.
- Change summary: no policy edit before this evaluation.
- Result: KEEP as the initial stage-3 incumbent only. `suite_average_gap=1.000000` on `cvrp_curriculum_12`; replay-10 stayed `0.047210`; replay-8 stayed `0.091085`.
- Per-instance gaps: train `cvrp_12_s0=1.000000`; replay `cvrp_10_s0=0.033195`; replay `cvrp_10_s1=0.061224`; replay `cvrp_8_s0=0.000000`; replay `cvrp_8_s1=0.182171`.
- Wall time: train suite wall time `441.9s`; replay-10 wall time `208.6s`; replay-8 wall time `62.1s`; harness wall time `714.2s`.
- Feasibility notes: `cvrp_12_s0` was infeasible with `raw_feasibility_rate=0.0` on 36 qubits and 144 parameters. The 10- and 8-customer replays remained feasible, so the failure is specific to 12-customer scaling.
- Next experiment: keep tilted capacity and test `angle_spread` Fisher-Jaikumar seed selection, because the 12-customer failure may come from depot-farthest seed geometry rather than the capacity penalty or final sampling.

## Iteration 19 - Stage 3 Angle-Spread Seeds

- Hypothesis: Angle-spread seeds may distribute the three Fisher-Jaikumar cluster centers more evenly across the 12-customer angular layout, making capacity-feasible tilted GAP assignments easier to sample than depot-farthest seeds.
- Failure signal: With depot-farthest seeds, `cvrp_12_s0` had zero feasible samples despite tilted capacity and 16384 final samples.
- Degrees of freedom exercised: CVRP construction seed method only: switch `cvrp_seed_method` from `depot_farthest` to `angle_spread`; keep tilted capacity, EfficientSU2 depth 1, expectation measurement, COBYLA 150, `estimator_shots=2048`, `sampler_shots=16384`, classical route solving, `seed=17`, and local search disabled.
- Change summary: set the adaptive CVRP construction default and CVRP policy field to `angle_spread`.
- Result: DISCARD. `suite_average_gap=1.000000` on `cvrp_curriculum_12`; replay-10 improved to `0.000000`; replay-8 improved to `0.000000`.
- Per-instance gaps: train `cvrp_12_s0=1.000000`; replay `cvrp_10_s0=0.000000`; replay `cvrp_10_s1=0.000000`; replay `cvrp_8_s0=0.000000`; replay `cvrp_8_s1=0.000000`.
- Wall time: train suite wall time `438.1s`; replay-10 wall time `191.0s`; replay-8 wall time `64.7s`; harness wall time `694.9s`.
- Feasibility notes: angle-spread seeds made all 10- and 8-customer replay instances optimal but still produced zero feasible samples on 36-qubit `cvrp_12_s0`. Because the primary train gap tied the incumbent at `1.000000`, the harness discarded it despite better replay.
- Next experiment: test a compression-oriented GAP solver on the 12-customer tilted formulation, because seed geometry alone did not make the 36-qubit VQE train sample feasible.

## Iteration 20 - Stage 3 QRAO GAP Compression

- Hypothesis: The 36-qubit tilted GAP VQE state still yields zero feasible samples; QRAO compression may reduce the effective quantum state dimension enough to recover feasible rounded assignments on `cvrp_12_s0`.
- Failure signal: Both depot-farthest and angle-spread tilted VQE had `raw_feasibility_rate=0.0` on `cvrp_12_s0`.
- Degrees of freedom exercised: GAP solver family only: switch CVRP `choose_solver_family` from VQE to QRAO. Keep tilted capacity, depot-farthest seeds, route solver classical, `sampler_shots=16384`, `estimator_shots=2048`, one-attempt stopping, `seed=17`, and local search disabled.
- Change summary: return `qrao` for CVRP in `choose_solver_family`; use the existing QRAO policy branch for >32-variable problems.
- Result: DISCARD. `suite_average_gap=1.000000` on `cvrp_curriculum_12`; replay-10 regressed to `0.098416`; replay-8 improved to `0.000000`.
- Per-instance gaps: train `cvrp_12_s0=1.000000`; replay `cvrp_10_s0=0.033195`; replay `cvrp_10_s1=0.163636`; replay `cvrp_8_s0=0.000000`; replay `cvrp_8_s1=0.000000`.
- Wall time: train suite wall time `63.3s`; replay-10 wall time `385.3s`; replay-8 wall time `263.1s`; harness wall time `712.9s`.
- Feasibility notes: QRAO compressed `cvrp_12_s0` to 12 qubits and was much faster on the train instance, but still had `raw_feasibility_rate=0.0`. Lower-scale replay remained feasible but was slower than VQE overall and worse on replay-10.
- Next experiment: return to VQE and increase the tilted capacity penalty strength, because the remaining 12-customer failure appears capacity-feasibility related rather than solely state-dimension related.

## Iteration 21 - Stronger Tilted Capacity Penalty

- Hypothesis: The default tilted penalty may be too weak at 12 customers, allowing the VQE distribution to favor over-capacity assignments. Increasing `cvrp_tilted_kappa` should strengthen capacity pressure without adding hard-slack variables.
- Failure signal: Tilted VQE with `kappa=5.0` had zero feasible samples on `cvrp_12_s0`, while lower-scale tilted runs had substantial feasible-sample rates.
- Degrees of freedom exercised: CVRP tilted capacity scaling only: raise `cvrp_tilted_kappa` from 5.0 to 10.0. Keep VQE, EfficientSU2 depth 1, depot-farthest seeds, expectation measurement, COBYLA 150, `estimator_shots=2048`, `sampler_shots=16384`, classical route solving, `seed=17`, and local search disabled.
- Change summary: set both the adaptive CVRP policy field and adaptive CVRP construction default for `cvrp_tilted_kappa` to 10.0.
- Result: DISCARD. `suite_average_gap=1.000000` on `cvrp_curriculum_12`; replay-10 stayed `0.047210`; replay-8 stayed `0.091085`.
- Per-instance gaps: train `cvrp_12_s0=1.000000`; replay `cvrp_10_s0=0.033195`; replay `cvrp_10_s1=0.061224`; replay `cvrp_8_s0=0.000000`; replay `cvrp_8_s1=0.182171`.
- Wall time: train suite wall time `332.4s`; replay-10 wall time `152.4s`; replay-8 wall time `49.2s`; harness wall time `535.2s`.
- Feasibility notes: `cvrp_12_s0` remained infeasible with `raw_feasibility_rate=0.0`; increasing tilted kappa did not change the decoded solution quality on the replay suites.
- Next experiment: try the Taylor soft capacity penalty family, because changing tilted penalty strength alone did not produce feasible 12-customer samples.

## Iteration 22 - Taylor Capacity Penalty

- Hypothesis: Taylor capacity penalties may shape the 12-customer GAP objective differently from tilted penalties and produce feasible capacity-respecting assignments without returning to hard-slack variables.
- Failure signal: hard-slack, tilted `kappa=5.0`, and tilted `kappa=10.0` all had zero feasible samples on `cvrp_12_s0`.
- Degrees of freedom exercised: CVRP GAP capacity penalty family only: switch `cvrp_gap_penalty_method` from `tilted` to `taylor`. Keep depot-farthest seeds, VQE EfficientSU2 depth 1, expectation measurement, COBYLA 150, `estimator_shots=2048`, `sampler_shots=16384`, classical route solving, `seed=17`, and local search disabled.
- Change summary: set the adaptive CVRP construction default and policy field to `taylor`.
- Result: DISCARD. `suite_average_gap=1.000000` on `cvrp_curriculum_12`; replay-10 stayed `0.047210`; replay-8 regressed to `0.134564`.
- Per-instance gaps: train `cvrp_12_s0=1.000000`; replay `cvrp_10_s0=0.033195`; replay `cvrp_10_s1=0.061224`; replay `cvrp_8_s0=0.086957`; replay `cvrp_8_s1=0.182171`.
- Wall time: train suite wall time `334.0s`; replay-10 wall time `154.8s`; replay-8 wall time `44.6s`; harness wall time `534.3s`.
- Feasibility notes: `cvrp_12_s0` remained infeasible with `raw_feasibility_rate=0.0`. Taylor kept the 10-customer replay feasible but worsened the 8-customer replay, so it offered no evidence of solving the 12-customer scaling failure.
- Next experiment: pause stage-3 edits unless a new hypothesis can directly address the zero-feasible-sample failure on `cvrp_12_s0` without sacrificing the confirmed 8- and 10-customer gains.

## Held-Out Final - E-n13 Benchmark

- Hypothesis: The locked tilted-capacity EfficientSU2 policy should be evaluated once on the reserved `E-n13-k4` benchmark after exceeding the 15-iteration search requirement; this result is for final reporting, not for keep/discard tuning.
- Failure signal: Stage-3 `cvrp_12_s0` still produced zero feasible samples, so the held-out final may also fail at 12-customer scale, especially with a different vehicle count.
- Degrees of freedom exercised: none; no policy edit before the final benchmark.
- Change summary: evaluate the current committed policy without changing `experiment.py`.
- Result: FINAL. `test_suite_average_gap=1.000000` on `cvrp_benchmark_e13`.
- Per-instance gaps: `cvrp_file_E-n13-k4=1.000000`.
- Wall time: held-out test wall time `389.0s`.
- Feasibility notes: the final `E-n13-k4` benchmark was infeasible with `raw_feasibility_rate=0.0`, `raw_ar=0.0`, 48 qubits, 192 parameters, 150 optimizer iterations, and one VQE attempt. This matches the stage-3 scaling failure mode and was not used for any further tuning.
- Next experiment: none under the current search contract; the locked policy improves 8- and 10-customer curriculum performance but does not solve the reserved 12-customer final benchmark.

## Redo Stage 3 - Fresh 12-Customer Search

- Hypothesis: A fresh 12-customer search should prioritize solver families and construction choices not already exhausted in the first stage-3 pass, while keeping the held-out benchmark reserved until after a candidate is locked.
- Failure signal: The locked tilted VQE policy, angle-spread seeds, QRAO compression, stronger tilted penalty, and Taylor capacity penalty all produced `raw_feasibility_rate=0.0` on `cvrp_12_s0`.
- Degrees of freedom exercised: none for this section header; this resets the search plan for `cvrp_curriculum_12` only.
- Change summary: use train-only 12-customer probes for likely-failing candidates, then run the normal stage-3 scout with replay guardrails if a candidate gets below `suite_average_gap=1.000000`.

## Redo Iteration 1 - PCE GAP Solver

- Hypothesis: PCE may recover a feasible 12-customer GAP assignment by optimizing a compressed MaxCut embedding of the tilted GAP QUBO, avoiding the 36-qubit VQE distribution that repeatedly sampled no feasible assignments.
- Failure signal: VQE and QRAO both had zero feasible sampled assignments on `cvrp_12_s0`; PCE has not yet been tested in the stage-3 search.
- Degrees of freedom exercised: GAP solver family only: switch CVRP from VQE to PCE. Keep tilted capacity, depot-farthest seeds, classical route solving, `sampler_shots=16384`, `estimator_shots=2048`, `seed=17`, and local search disabled.
- Change summary: return `pce` for CVRP in `choose_solver_family`; use the existing PCE policy branch.
- Result: DISCARD. Train-only `cvrp_curriculum_12` probe stayed at `suite_average_gap=1.000000`.
- Per-instance gaps: train `cvrp_12_s0=1.000000`.
- Wall time: train suite wall time `15.5s`.
- Feasibility notes: PCE was fast but still produced an infeasible decoded GAP assignment. Because the primary train metric tied the incumbent at the failure value, no replay scout was run.
- Next experiment: return to VQE and test a new Fisher-Jaikumar seed method, because PCE did not address the assignment-feasibility failure.

## Redo Iteration 2 - Largest-Demand Seeds

- Hypothesis: The 12-customer failure may be driven by the VQE distribution missing a small feasible basin. On `cvrp_12_s0`, largest-demand seeds produce more capacity-feasible seed-constrained assignments than depot-farthest or angle-spread, so they may make any feasible decoded assignment easier to sample even if the best exact Fisher-Jaikumar route is worse.
- Failure signal: Depot-farthest VQE had zero feasible samples; angle-spread improved lower-scale replay but still had zero feasible samples on the 12-customer train instance; PCE also stayed infeasible.
- Degrees of freedom exercised: CVRP seed method only: switch `cvrp_seed_method` from `depot_farthest` to `largest_demand`. Keep tilted capacity, VQE EfficientSU2 depth 1, expectation measurement, COBYLA 150, `estimator_shots=2048`, `sampler_shots=16384`, classical route solving, `seed=17`, and local search disabled.
- Change summary: set both the adaptive CVRP construction default and policy field to `largest_demand`.
- Result: DISCARD. Train-only `cvrp_curriculum_12` probe stayed at `suite_average_gap=1.000000`.
- Per-instance gaps: train `cvrp_12_s0=1.000000`.
- Wall time: train suite wall time `331.3s`.
- Feasibility notes: largest-demand seeds did not produce a feasible decoded assignment despite the larger capacity-feasible assignment count in the seed-constrained enumeration. Because the primary train metric tied the incumbent at the failure value, no replay scout was run.
- Next experiment: restore depot-farthest seeds and inspect the tilted capacity objective before changing another runtime-heavy VQE knob.

## Redo Iteration 3 - Lower Tilted Penalty Scale

- Hypothesis: The tilted objective's exact seed-constrained optimum is capacity-feasible for `cvrp_12_s0`, so the failure may be VQE conditioning rather than an infeasible soft-penalty optimum. Reducing `cvrp_tilted_kappa` may shrink coefficient scale and improve the sampled state.
- Failure signal: Increasing `cvrp_tilted_kappa` to 10.0 did not help in the prior stage-3 pass; direct objective enumeration showed the best tilted assignment remains feasible even at lower kappa values.
- Degrees of freedom exercised: CVRP tilted capacity scaling only: lower `cvrp_tilted_kappa` from 5.0 to 0.5. Keep VQE, EfficientSU2 depth 1, depot-farthest seeds, expectation measurement, COBYLA 150, `estimator_shots=2048`, `sampler_shots=16384`, classical route solving, `seed=17`, and local search disabled.
- Change summary: set both the adaptive CVRP policy field and adaptive CVRP construction default for `cvrp_tilted_kappa` to 0.5.
- Result: DISCARD. Train-only `cvrp_curriculum_12` probe stayed at `suite_average_gap=1.000000`.
- Per-instance gaps: train `cvrp_12_s0=1.000000`.
- Wall time: train suite wall time `334.9s`.
- Feasibility notes: lowering the tilted coefficient scale did not produce a feasible decoded assignment. Because the primary train metric tied the incumbent at the failure value, no replay scout was run.
- Next experiment: restore `cvrp_tilted_kappa=5.0` and test a less aggressive QRAO compression than the previous 3-to-1 stage-3 probe.

## Redo Iteration 4 - QRAO 2-to-1 GAP Compression

- Hypothesis: Prior QRAO used 3-to-1 compression on the 36-variable tilted GAP and still produced no feasible samples. A 2-to-1 QRAO encoding may preserve more GAP assignment structure while remaining smaller than full 36-qubit VQE.
- Failure signal: Full VQE remains infeasible under multiple construction tweaks, and the previous 12-qubit QRAO 3-to-1 candidate was fast but too lossy.
- Degrees of freedom exercised: GAP solver family and QRAO compression ratio: switch CVRP to QRAO with `qrao_max_vars_per_qubit=2`, `qrac_type=2`, semideterministic rounding, real-amplitudes depth 1. Keep tilted capacity, depot-farthest seeds, `sampler_shots=16384`, `estimator_shots=2048`, classical route solving, `seed=17`, and local search disabled.
- Change summary: return `qrao` for CVRP in `choose_solver_family` and add a CVRP-specific QRAO policy branch using 2-to-1 encoding.
- Result: DISCARD. Train-only `cvrp_curriculum_12` probe stayed at `suite_average_gap=1.000000`.
- Per-instance gaps: train `cvrp_12_s0=1.000000`.
- Wall time: train suite wall time `215.9s`.
- Feasibility notes: the 18-qubit QRAO encoding was slower than the previous 12-qubit 3-to-1 QRAO probe and still produced an infeasible decoded assignment. Because the primary train metric tied the incumbent at the failure value, no replay scout was run.
- Next experiment: keep the 2-to-1 QRAO encoding but switch rounding to magic to sample a broader set of rounded assignments.

## Redo Iteration 5 - QRAO 2-to-1 Magic Rounding

- Hypothesis: Semideterministic QRAO rounding may collapse the 2-to-1 encoding to one infeasible assignment. Magic rounding can sample a broader rounded-solution distribution and may recover a feasible GAP assignment without returning to full 36-qubit VQE.
- Failure signal: Both QRAO 3-to-1 semideterministic and QRAO 2-to-1 semideterministic stayed infeasible on `cvrp_12_s0`.
- Degrees of freedom exercised: QRAO rounding only: keep `qrao_max_vars_per_qubit=2`, `qrac_type=2`, real-amplitudes depth 1, and switch `rounding` from `semideterministic` to `magic`. Keep tilted capacity, depot-farthest seeds, `sampler_shots=16384`, `estimator_shots=2048`, classical route solving, `seed=17`, and local search disabled.
- Change summary: set the CVRP QRAO branch rounding to `magic`.
- Result: DISCARD. Train-only `cvrp_curriculum_12` probe stayed at `suite_average_gap=1.000000`.
- Per-instance gaps: train `cvrp_12_s0=1.000000`.
- Wall time: train suite wall time `470.1s`.
- Feasibility notes: magic rounding broadened the QRAO decode path but still produced an infeasible assignment, and it was slower than both 3-to-1 QRAO and 2-to-1 semideterministic QRAO. Because the primary train metric tied the incumbent at the failure value, no replay scout was run.
- Next experiment: return to the confirmed VQE tilted/depot policy and increase only final sampling to test whether feasible support exists below the 16384-shot observation threshold.

## Redo Iteration 6 - High Final Sampling On Tilted VQE

- Hypothesis: The confirmed tilted/depot VQE state may contain feasible 12-customer assignments below the 16384-shot observation threshold. Increasing final sampler shots may recover at least one feasible decode without changing the optimizer trajectory.
- Failure signal: The tilted objective's exact optimum is feasible, but the current VQE final samples have observed zero feasible assignments at 16384 shots.
- Degrees of freedom exercised: final sampling only: raise CVRP `sampler_shots` from 16384 to 65536. Keep VQE EfficientSU2 depth 1, tilted capacity, depot-farthest seeds, expectation measurement, COBYLA 150, `estimator_shots=2048`, classical route solving, `seed=17`, and local search disabled.
- Change summary: restore CVRP to VQE and set `sampler_shots=65536`.
- Result: DISCARD by runtime. The train-only `cvrp_curriculum_12` probe was manually stopped after exceeding the 900s instance budget without a completed summary.
- Per-instance gaps: no completed train split summary was produced.
- Wall time: stopped after more than 16 minutes.
- Feasibility notes: increasing final samples to 65536 made the 12-customer probe too expensive for the scratch loop before it could report feasibility. This is not a viable path unless a faster optimizer state is found first.
- Next experiment: restore `sampler_shots=16384` and test a bounded CVaR objective with reduced estimator shots and optimizer iterations, because the expectation-optimized state has not exposed feasible samples.

## Redo Iteration 7 - Bounded CVaR VQE

- Hypothesis: Expectation VQE may optimize average energy while leaving feasible assignments out of sampled support. A bounded CVaR objective may bias the optimizer toward lower-energy sampled tails and expose a feasible 12-customer assignment, while reduced estimator shots and max iterations keep the probe within budget.
- Failure signal: Multiple expectation-VQE construction changes and high final sampling failed or became too slow; prior CVaR on the original hard-slack 8-customer setup was too slow, so this test must reduce the CVaR budget.
- Degrees of freedom exercised: VQE measurement objective and optimizer budget: switch CVRP VQE to `measurement_mode="cvar"`, `cvar_alpha=0.10`, `estimator_shots=512`, and `optimizer_maxiter=60`. Keep EfficientSU2 depth 1, tilted capacity, depot-farthest seeds, `sampler_shots=16384`, classical route solving, `seed=17`, and local search disabled.
- Change summary: restore `sampler_shots=16384`; set CVRP VQE to bounded CVaR with lower estimator shots and fewer COBYLA iterations.
- Result: DISCARD. Train-only `cvrp_curriculum_12` probe stayed at `suite_average_gap=1.000000`.
- Per-instance gaps: train `cvrp_12_s0=1.000000`.
- Wall time: train suite wall time `300.6s`.
- Feasibility notes: bounded CVaR completed within budget and used the expected CVaR policy path, but still produced an infeasible decoded assignment. Because the primary train metric tied the incumbent at the failure value, no replay scout was run.
- Next experiment: restore expectation VQE and test the tilted penalty shape parameter `cvrp_tilted_s_frac`, because objective-tail optimization did not expose feasible samples.

## Redo Iteration 8 - Wider Tilted Capacity Target

- Hypothesis: Adjusting `tilted_s_frac` changes the tilted capacity penalty shape without adding hard-slack variables. A wider tilted target may improve VQE conditioning around balanced capacity-feasible assignments on `cvrp_12_s0`.
- Failure signal: kappa scaling alone did not help, but `s_frac` changes a different part of the tilted penalty. Direct objective enumeration showed depot-farthest tilted optima remain capacity-feasible across wider `s_frac` values.
- Degrees of freedom exercised: CVRP tilted shape only: raise `cvrp_tilted_s_frac` from 0.10 to 0.20. Keep VQE EfficientSU2 depth 1, tilted capacity, depot-farthest seeds, expectation measurement, COBYLA 150, `estimator_shots=2048`, `sampler_shots=16384`, classical route solving, `seed=17`, and local search disabled.
- Change summary: restore the confirmed expectation-VQE budget and set both the adaptive CVRP policy field and construction default for `cvrp_tilted_s_frac` to 0.20.
- Result: DISCARD. Train-only `cvrp_curriculum_12` probe stayed at `suite_average_gap=1.000000`.
- Per-instance gaps: train `cvrp_12_s0=1.000000`.
- Wall time: train suite wall time `332.5s`.
- Feasibility notes: widening the tilted capacity target did not produce a feasible decoded assignment. Because the primary train metric tied the incumbent at the failure value, no replay scout was run.
- Next experiment: stop the fresh 12-customer search without a new keep, restore the confirmed stage-2 policy, and rerun the held-out benchmark once for reporting.

## Redo Held-Out Final - E-n13 Benchmark

- Hypothesis: No fresh 12-customer candidate improved on the locked policy, so the benchmark rerun should use the restored confirmed tilted/depot VQE policy and is expected to remain difficult. This is a reporting rerun, not a keep/discard tuning step.
- Failure signal: Every fresh 12-customer train probe in the redo phase stayed at `suite_average_gap=1.000000` or was discarded by runtime.
- Degrees of freedom exercised: none; restore the locked policy before benchmark evaluation.
- Change summary: evaluate the current restored policy without changing `experiment.py`.
- Result: FINAL RERUN. `test_suite_average_gap=1.000000` on `cvrp_benchmark_e13`.
- Per-instance gaps: `cvrp_file_E-n13-k4=1.000000`.
- Wall time: held-out test wall time `386.6s`.
- Feasibility notes: the benchmark rerun was infeasible with `raw_feasibility_rate=0.0`, `raw_ar=0.0`, 48 qubits, 192 parameters, 150 optimizer iterations, and one VQE attempt. This matches the earlier held-out final result and was not used for any further tuning.
- Next experiment: none in this redo phase; the fresh 12-customer candidates did not beat the locked policy.

## Hybrid Final Protocol - Stage 2 E-n13 Final

- Timestamp: 2026-04-12 10:49:07 +08.
- Hypothesis: The hybrid reduced-GAP solver should produce a feasible E-n13 result where the restored full-space VQE policy previously failed.
- Failure signal: Prior E-n13 final rerun had `suite_average_gap=1.000000`, `raw_feasible=0`, and zero observed feasible raw GAP samples.
- Degrees of freedom exercised: no new policy edit in this step; run the program-specified `cvrp_benchmark_e13` final workflow with local search disabled.
- Change summary: none; held-out final evaluation only.
- Result: FINAL. Test `suite_average_gap=0.139373`.
- Per-instance gaps: `cvrp_file_E-n13-k4=0.139373`.
- Wall time: held-out test suite wall time `30.0s`.
- Feasibility notes: all feasible; one hybrid attempt, AR `0.861`, policy summary `HYBRID efficientsu2 d=1 COBYLA`.
- Next experiment: keep held-out work limited to E-n13 unless the user explicitly expands the benchmark scope.

## Hybrid Final Protocol - 12-Customer Verification Rerun

- Timestamp: 2026-04-12 10:55:54 +08.
- Hypothesis: The hybrid solver plus feasibility repair should produce a feasible replacement result for the 12-customer training instance, and this rerun should be recorded in the CVRP artifacts rather than run with `--no-artifacts`.
- Failure signal: The prior verification command used `--no-artifacts`, so it did not replace the logged CVRP suite result even though it completed successfully.
- Degrees of freedom exercised: no policy edit; rerun `cvrp_curriculum_12` train split with artifacts enabled and local search disabled.
- Change summary: removed the no-artifact verification journal entry and reran the recorded 12-customer evaluation.
- Result: KEEP for proceeding to E-n13-only held-out final. Train `suite_average_gap=0.010101`.
- Per-instance gaps: `cvrp_12_s0=0.010101`.
- Wall time: train suite wall time `4.7s`.
- Feasibility notes: all feasible; one hybrid attempt, AR `0.990`, policy summary `HYBRID efficientsu2 d=1 COBYLA`, eval group `32`.
- Next experiment: run only the E-n13 held-out final benchmark.

## Hybrid Final Protocol - E-n13-Only Held-Out Rerun

- Timestamp: 2026-04-12 10:56:55 +08.
- Hypothesis: With held-out scope restricted to E-n13, the hybrid reduced-GAP solver should reproduce the feasible E-n13 result without running any additional benchmark suite.
- Failure signal: The previous held-out protocol was too broad for the requested scope; this rerun keeps only E-n13.
- Degrees of freedom exercised: no policy edit; rerun only `cvrp_benchmark_e13` final workflow with local search disabled.
- Change summary: held-out scope reduced to E-n13-only in the runbook and suite definitions; no solver change.
- Result: FINAL. Test `suite_average_gap=0.139373`.
- Per-instance gaps: `cvrp_file_E-n13-k4=0.139373`.
- Wall time: held-out test suite wall time `28.9s`.
- Feasibility notes: all feasible; one hybrid attempt, AR `0.861`, policy summary `HYBRID efficientsu2 d=1 COBYLA`, eval group `33`.
- Next experiment: none unless the user requests a new CVRP experiment.
