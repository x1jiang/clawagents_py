# ATLAS A/B experiment (`gpt-5-nano`)

| task | atlas | ok | iters | tools | sec | tokens | transcript | ckpts | gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| easy_list | off | ✓ | 2 | 1 | 10.5 | 11106 | 7256 | 0 | 0 |
| easy_list | on | ✓ | 4 | 1 | 68.0 | 43688 | 29193 | 3 | 7 |
| fail_then_fix | off | ✓ | 2 | 1 | 12.0 | 11642 | 8311 | 0 | 0 |
| fail_then_fix | on | ✓ | 12 | 8 | 121.2 | 132216 | 37820 | 15 | 3 |
| verify_before_done | off | ✓ | 3 | 2 | 9.9 | 15904 | 7206 | 0 | 0 |
| verify_before_done | on | ✓ | 4 | 2 | 27.8 | 28004 | 23098 | 1 | 5 |

## Pairwise deltas (on − off)

| task | token Δ | time Δ | transcript Δ | iter Δ | ckpts | gates | ok off→on |
| --- | --- | --- | --- | --- | --- | --- | --- |
| easy_list | 32582 | 57.48 | 21937 | 2 | 3 | 7 | True→True |
| fail_then_fix | 120574 | 109.22 | 29509 | 10 | 15 | 3 | True→True |
| verify_before_done | 12100 | 17.95 | 15892 | 1 | 1 | 5 | True→True |

## Task expectations
- **easy_list**: should be cheap; ATLAS overhead = final gate only
- **fail_then_fix**: ATLAS should fire on tool failure; may reduce blind retries
- **verify_before_done**: final gate may catch premature DONE without verify
