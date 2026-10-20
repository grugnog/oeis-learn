# Diverse32 memorization gate — T096

Status: pending the actual HIP/data/cost gates. No training run was launched by
the agent because this environment has no container engine or HIP device.

`--gpu --prepare --learnability` in the Phase 5 helper requests the bounded
registered smoke run only after preceding gates pass. The original recipe remains
d256,4+4 layers, batch8, LR1e-4, FP32/eager, seed20261016, up to3000 updates/7200s,
100-update validation, visible-only selection and the frozen patience rule.
The actual prepared tiny set must contain32 prompts with the required structure,
offset and non-affine composition. A raw runtime fixture cannot replace it.

The public learnability path records raw/rescue, correct/shuffled/absent and
non-neural controls, independently scored100-index continuations, termination,
coverage and the selected real checkpoint. The unchanged gate requires at least
31/32 rescue successes,32/32 complete evidence and32/32 greedy termination.
The helper records PASS or FAIL from that gate and returns BLOCKED on a negative
outcome, retaining the run bundle. It never lowers the threshold, changes seed,
starts representation screens or evaluates the sealed final cohort automatically.
