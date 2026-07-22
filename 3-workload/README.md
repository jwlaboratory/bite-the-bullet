# Workload

Everything about the workload the `early_rdma` policy targets: a synchronized
fan-out of requests sharing a long, job-unique prefix (a data-labeling sweep, or
an agent fanning out to ~20 sub-agents that all inherit the same big context).

- [`audit/`](audit/) — **does this pattern exist in public traces?** Streams
  ART, Mooncake, BurstGPT, etc. and measures same-prefix co-arriving fan-out.
  The answer is mostly *no*, which is why we synthesize it.
- [`generate/`](generate/) — **builds Bursted-ART**, the synthetic workload that
  *does* contain the pattern. Published at
  [shreybirmiwal/Bursted-ART](https://huggingface.co/datasets/shreybirmiwal/Bursted-ART).
