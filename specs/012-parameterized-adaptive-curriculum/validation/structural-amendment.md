# Structural feasibility amendment — 2026-11-05

Authorization: after reviewing the uploaded failed admission, the operator requested both the cohort blocker and missing nested-cause reporting be fixed. This is an explicit new scientific scope, frozen before any new model inference. It does not turn the failed original scope into a PASS.

## Evidence and limits

Original public archive: `sha256:df351d22447a500681aaef83a3a3923b4ab339e74bb51ecccc12cf6eed8122fa`. Availability: `sha256:520fa8b8b5fce79548a0ebdad1b754f7225723cbdbe44db30d3deadda7526ba0`. The old 2,288 draws leave at most 53/68 certified and 17/26 ambiguous slots filled under their fixed family quotas; four families have no survivors. All 894 excluded draws have a finite-window collision. The archive contains native completion counters, not a completed native-admission certificate.

Coefficient broadening cannot repair a family whose parameter-independent 20-term branch window already occurs in the original corpus. We preserve the conservative exclusion rule rather than claiming such windows are novel.

## Frozen replacement

`pac-structural-scope/v2` retains exactly the original 2,288 candidate identities/order, seed, coefficients/numeric law, all sixteen original exclusion inputs and every source/group/numeric/window/family exclusion. No draw is added or reseeded. The training pool, train-only probes, learner policy, initialization, 25M-token allowances, 20/100 terms, eight attempts, 60-second portfolios, seven controls and 65/68 completion floor are unchanged.

The revised population has 128 tasks across **23 families**: **68 certified tasks/15 families**, **26 ambiguous/3 families**, **34 unresolved/5 families**. It estimates performance conditional on this feasibility-screened population. It is not a result for the old 27-family/18-certified-family population and must not be compared with its aggregate scores. Every compared checkpoint must be scored again on the same new frozen roster. Reference-only A remains reference-only.

Quota construction is deterministic and uses only public admission counts plus the already-frozen candidate-source multiplicities. For each family, sort source multiplicities descending and find the fewest distinct sources that could contain its reported survivor count. This lower bound protects against the worst possible survivor duplication. Clamp each old quota to that bound, then distribute missing slots one at a time in ascending family-hash order within the same category, respecting every bound. Freeze the resulting table; never reallocate after the next admission or model outcomes. There are no concrete-source overlaps between candidate families. Thus every survivor subset with the reported per-family counts has enough distinct sources for these quotas. This is a capacity proof, not a substitute for fresh native agreement and original-input verification.

Amendment identity: `sha256:d7d5b559cb2247500129440480be82646d972ce20b58d7e279650f8501ccc81c`. The source-bound policy resource records full hashes, capacities, exact quotas and failed-evidence references. A new plan/image/source closure is required; the old plan cannot be resumed into this population.

| Family hash prefix | Old quota | Reported survivors | Guaranteed distinct sources | New quota |
| --- | ---: | ---: | ---: | ---: |
| `023c94524b41` | 4 | 64 | 64 | 4 |
| `04986e9882b8` | 2 | 62 | 62 | 4 |
| `0aa27908fe64` | 2 | 0 | 0 | 0 |
| `174a9a2071d5` | 4 | 63 | 63 | 6 |
| `286588d220a6` | 6 | 95 | 95 | 11 |
| `322d33b6b985` | 6 | 0 | 0 | 0 |
| `34063c915043` | 7 | 0 | 0 | 0 |
| `3f7f434b8e62` | 2 | 4 | 2 | 2 |
| `589833f0edcd` | 4 | 37 | 36 | 6 |
| `5c475cecf9da` | 4 | 1 | 1 | 1 |
| `7d3c681969ef` | 4 | 60 | 59 | 5 |
| `814c9514aa68` | 1 | 7 | 7 | 2 |
| `83a8423db5b0` | 4 | 2 | 2 | 2 |
| `988a472f4893` | 6 | 22 | 22 | 7 |
| `ae2291fc1dea` | 5 | 74 | 74 | 5 |
| `cf0cf185ed7e` | 2 | 6 | 6 | 3 |
| `d2a0230d8ec5` | 4 | 64 | 64 | 4 |
| `d5849ec7e32b` | 10 | 155 | 155 | 11 |
| `daaacba51cb1` | 10 | 160 | 160 | 10 |
| `dbeda6df0cf5` | 11 | 176 | 176 | 11 |
| `df92fd60c4a5` | 9 | 131 | 131 | 13 |
| `e0ba7e3a2bfe` | 3 | 64 | 64 | 4 |
| `eb82f1176d0d` | 3 | 9 | 9 | 4 |
| `ee89096d39c3` | 4 | 2 | 2 | 2 |
| `f16e6cc78e4e` | 2 | 0 | 0 | 0 |
| `f6bb7c891347` | 5 | 76 | 76 | 6 |
| `fe4b4781a7b4` | 4 | 60 | 60 | 5 |

## Compatibility and validation

Historical `pac-structural-scope/v1`, its 27/18-family metric guards, original failures and 011 records remain unchanged. New admission/scoring/report records bind v2 by exact design identity and exact family quotas. A mismatched or unknown scope cannot qualify or be silently interpreted as v1. Runtime admission still blocks any deficit or native disagreement.

Independent local regressions reconstruct worst-case duplicate survivor subsets from the uploaded counters: v1 fails, v2 admits exactly its declared 128/23 population, both scopes use byte-identical candidate draws, and v1 metrics reject a v2 roster. Actual new original-input/native/HIP acceptance remains pending; no research or HIP command is launched by this repair.
