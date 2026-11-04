# Generalization contract fixtures

Run `python3 tests/fixtures/generalization/generate.py` to reproduce the JSON payloads and inventory. The generator uses only standard-library arithmetic and formatting; it never imports project admission, fitting, grouping or selection code.

Authority examples have synthetic hashes and flags solely to exercise JSON shape. They are not actual qualified/admitted records. Mechanical source cases have independent closed-form expected terms. Policy expectations are golden counterexamples for Phase B, not a Phase A implementation of those policies. See `specs/011-controlled-generalization/contracts/algorithm-handoff.md` for ownership and exit criteria.

## PAC policy fixtures (012 T003)

`pac-policy.json` is transcribed from hand arithmetic, not emitted by the kernel.
With uniform thirds, eta=log(2)/3 and selected p=1/3, the rewarded weight doubles
(or halves for -1). Graves' first alpha=1/2 shares half of each *other* weight,
then normalizes: positive q=(3/8,5/16,5/16), negative q=(3/10,7/20,7/20).
Mixing 3/4 q + 1/12 yields the stored fractions. At alpha=1/4, zero-reward
q=(3/5,3/10,1/10) becomes (1/2,5/16,3/16); missing has no sharing.
Queue due times are committed exposure + 512*2^queue. Incidental practice resets
the clock without changing level. Fault expectations use batches of exactly 17
charged tokens, with eight successful updates required for one complete reward.
Structural totals transcribe the frozen 011 allocation, not measured admission.
The earlier planning simulations are not production test evidence.
