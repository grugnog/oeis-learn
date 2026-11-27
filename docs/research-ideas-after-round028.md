# Research discussion after round028

**Status: working notes for a future spec.** The measurements below were published. The experiments below them were discussed afterward and have no registration, budget or launch authority.

## Where the discussion started

- The historical CPU audit reopened 128 exposed development cases. On the 64 composition and unseen-structure cases used for the decision, trial 05 had seven selected solves, six unresolved cases and 51 completed searches in which every admitted program missed despite an independently verified solution. It found zero confirmed cases where the selector passed over a correct answer. This establishes a gap in that search on those cases. A witness can use information unavailable to a learner seeing the first 20 terms. [Audit](../specs/015-learned-program-search/validation/round028-audit.md)
- In round028’s one-seed pilot, both arms trained on 4,000,210 useful tokens in 2,126 updates. On 128 development and 32 retention cases, ordinary training solved 22 + 4, first-divergence training solved 13 + 4, and retained D solved 21 + 6. On the primary family-weighted comparison, first divergence trailed ordinary training by 14.17 percentage points. The missing-outcome bounds were −22.83 to −6.50 points. Ordinary versus D remained inconclusive. D was retained and final assessment stayed sealed. [Verified pilot](../specs/015-learned-program-search/validation/round028-pilot.md)
- Charged time for the whole pilot was 1 h 56 min 56 s, including about 65 minutes of search and 52 minutes of training across both arms. The 0 and 1M looks used one attempt on a subset. The full 4M endpoint used eight attempts. A velocity comparison requires a common probe across checkpoints. Hardware recovery, source and installed-wheel checks have their own evidence. [Pilot timing](../specs/015-learned-program-search/validation/round028-pilot.md#forecast-and-charged-time) · [Release results](../specs/015-learned-program-search/validation/round028-release.md)

## Two ideas to consider independently

**Explore which variables improve training velocity.** The round028 ordinary arm is a potential historical training control. Round027’s trial 00 was a different recipe and stopped at 2M. A suggested initial design had two groups of four recipes sharing the ordinary control, for seven distinct recipes in all.

| Group | Proposed comparison |
| --- | --- |
| Training data and conditioning | Cross the existing data mix versus broader verified structural compositions with ordinary supervision versus an auxiliary exercise matching a program to its genuine prefix and distinguishing it from a similar prefix it demonstrably fails. Use training-only families. |
| Optimizer | Cross effective batch 16/32 with weight decay 0/0.01, holding the learning rate and data recipe fixed. Historical trial 05 changed both settings together. The ordinary arm supplies the batch-32, zero-decay cell when the inputs match. |

Measure paired exact yield at comparable token counts **and** workstation time. Retain unresolved cases and retention results. A 1M reading can identify a broken run; the proposed shared 4M endpoint would allow slower starters to be seen. Re-score preserved checkpoints if the probe changes. A favorable exploratory result could justify an independent seed and fresh families. If a longer run is warranted, continue both the finalist and its ordinary control. These are candidate choices for discussion.

**Test whether the sequence can guide program construction.** On separate synthetic practice families, give the existing fitter the known program structure with its constants removed. This would separate problems within its reach from failures of fitting or ambiguity in the visible prefix. A second experiment could compare fixed-order structural search with a learned policy that proposes the next operation from the visible sequence and an executable partial result. Include D and a shuffled-model-input control, keep the fitter on the true prefix, and price internal exploration as part of the total search allowance. Track both the presence of any correct proposal and the exact continuation of the single selected program.

The two directions answer different questions. Faster training requires curves at comparable exposure and cost. Better search can be tested with a fixed model or a separately measured guide. Before writing a protocol, decide which result would change the research decision, which existing outputs can genuinely be reused, and which families remain unexposed. [Evidence and workstation file index](../specs/015-learned-program-search/research.md#finding-the-evidence-without-the-conversation)
