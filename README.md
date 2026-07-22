# Bite the Bullet

Detect sustained shared-prefix reuse, then RDMA-copy existing KV into HBM on
less-busy replicas *before* later requests arrive — while the fabric is still
quiet — instead of recomputing the prefix or contending for it reactively during
the spike. The active policy is `early_rdma`.


1. workload/audit: does workload even exist in public traces
2. workload/generate: synthetic generate the bursted-art dataset
3. experiments/1-prefill-vs-transfer: moving kv beats recomputing
4. experiments/2-burst-routing: 
5. experiments/3-early-rdma: 


# Dependency

Every live experiment runs on a sibling `inference-sim` checkout:

```text
../inference-sim
```

Override with:

```bash
export INFERENCE_SIM_ROOT=/path/to/inference-sim
```

Live experiment scripts assume they sit exactly two directories deep
(`experiments/<name>/script.py`) and that `sim_path.py` stays at the repo
root — keep that shape when adding experiments.



# Future ideas

1. cost aware gate: trigger only when reuse beats movement cost and tail risks
2. better network model for fabric congestion
3. test on real serving stack
4. partial-prefix warming;
5. fake-prefill warming;
6. seed-real warming;
7. learned router / online router;
8. dynamic re-sharding.
