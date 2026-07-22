# Bite the Bullet

Detect a sustained shared-prefix burst, then RDMA-copy its KV onto less-busy
replicas **before** later requests arrive — so those requests land on a warm
node instead of recomputing the prefix. The policy is `early_rdma`.

The repo is three folders:

| # | Folder | What |
|---|--------|------|
| 1 | [`1-prefill-vs-transfer/`](1-prefill-vs-transfer/) | **The motivation.** Recomputing a prefix vs transferring its KV from HBM / RAM / RDMA / disk. Prefill costs ~44× an RDMA transfer — that gap is what warming spends. |
| 2 | [`2-bite-the-bullet/`](2-bite-the-bullet/) | **The algorithm + its result.** One file with the four-constant `early_rdma` rule; replays the real Bursted-ART test set across model×hardware setups (70B, GLM, Qwen, Kimi, dense-1T). Cuts mean TTFT +11% to +70% vs the SGLang default router. |
| 3 | [`3-workload/`](3-workload/) | **The workload.** `generate/` builds the Bursted-ART burst dataset; `audit/` checks whether the pattern shows up in public traces (mostly it doesn't). |

Everything runs on the [Infer-Sim](https://jwlabs.vercel.app/post/infer-sim)
engine — clone `inference-sim` next to this repo, or set `INFERENCE_SIM_ROOT`.

```bash
python3 1-prefill-vs-transfer/compute_costs.py     # the cost gap
python3 2-bite-the-bullet/bite_the_bullet.py       # the algorithm + result
```

Superseded material is under [`archive/`](archive/).
