# Benchmark results

Rows appended by `benchmark.py`. Same seed list per run (default seeds 1..N) so agents are directly comparable.

| date | agent | config | model | games | @1024 | @2048 | @4096 | @8192 | @16384 | mean score | median | max tile | moves/game | moves/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-13 | n-tuple greedy | greedy(d=1) | ntuple_2048_1783948747 | 300 | 97.7% | 79.3% | 29.7% | 0.7% | 0.0% | 42119 | 39560 | 8192 | 2093 | 107594 |
| 2026-07-13 | expectimax n-tuple | depth=2 | ntuple_2048_1783948747 | 300 | 99.7% | 96.7% | 83.7% | 19.0% | 0.0% | 76211 | 75758 | 8192 | 3449 | 5098 |
| 2026-07-14 | expectimax n-tuple | adaptive[2:4,5:3,16:2] | ntuple_2048_1783948747 | 200 | 100.0% | 99.5% | 96.5% | 47.0% | 0.0% | 103861 | 82926 | 8192 | 4518 | 259 |
| 2026-07-14 | n-tuple greedy | greedy(d=1) | ntuple_2048_t8_tc_1783965407 | 300 | 100.0% | 96.0% | 78.3% | 23.7% | 0.0% | 81386 | 77548 | 8192 | 3657 | 181774 |
| 2026-07-14 | expectimax n-tuple | depth=2 | ntuple_2048_t8_tc_1783965407 | 300 | 100.0% | 98.7% | 93.7% | 68.3% | 3.7% | 128727 | 141250 | 16384 | 5443 | 17256 |
| 2026-07-14 | n-tuple greedy | greedy(d=1) | ntuple_2048_t4_tc_1783965407 | 300 | 99.0% | 92.7% | 73.0% | 23.3% | 0.3% | 77350 | 75314 | 16384 | 3479 | 220346 |
| 2026-07-14 | expectimax n-tuple | depth=2 | ntuple_2048_t4_tc_1783965407 | 300 | 100.0% | 99.3% | 96.3% | 62.7% | 2.0% | 123514 | 132158 | 16384 | 5259 | 17288 |
