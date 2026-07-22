# Future Ideas

Ideas worth keeping after the cleanup.

## 1. Ten-Minute Rising Hot Prefix

Create a controlled trace where one long prefix gets a rising request rate over
10 minutes.

Goal: show the ideal BTB regime where movement cost is fixed and future reuse
keeps growing.

Measure:

- mean TTFT;
- p95 TTFT;
- HBM hit rate;
- RDMA GB moved;
- time-to-payback.

## 2. Tail-Aware BTB Gate

The current trigger is a simple repeated-prefix rule. The next gate should
fire only when expected reuse beats movement cost and tail risk.

Inputs:

- recent same-prefix count;
- expected future reuse;
- HBM pressure;
- RDMA pressure;
- queue depth;
- model/hardware setup.

## 3. Better Network Model

The simulator currently charges per-target RDMA copy time. It does not model
shared fabric congestion.

Add:

- source NIC contention;
- switch/fabric contention;
- concurrent copy interference;
- per-link saturation.

## 4. Live Serving Validation

Reproduce the best `early_rdma` case on a real serving stack.

Candidates:

- vLLM prefix caching;
- SGLang prefix caching;
- Dynamo-style KV routing;
- small cluster first, then H100-class cluster.

## 5. Weight/KV Dtype Split

The simulator should model weight dtype and KV dtype separately.

Needed for GLM/Kimi-style setups where weights may be int4/fp8 but KV is fp8 or
fp16.

## 6. Archived Ideas To Revisit Carefully

These are archived, not active claims:

- partial-prefix warming;
- fake-prefill warming;
- seed-real warming;
- learned router / online router;
- dynamic re-sharding.

They may be useful later, but the current paper should stay focused on
`early_rdma`.
