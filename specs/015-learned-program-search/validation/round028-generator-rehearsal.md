# Matched generator training rehearsal

The archive-only foundation verifier reproduced the forecast from both arms' original update timings and token charges. The workstation verified all 13 files of each full-state checkpoint, including optimizer and RNG state. The installed learner changed parameters in each arm and reproduced the next update after cold restore.

| Observed on the original workstation | Ordinary | First divergence |
| --- | ---: | ---: |
| Positive tokens, first / second update | 2,060 / 1,836 | 2,060 / 1,836 |
| Preference rows in the first batch of 32 | 0 | 19 |
| First / second update, seconds | 2.151876 / 0.622875 | 1.068402 / 0.624861 |
| Save / restore, seconds | 2.106938 / 2.127572 | 2.185682 / 1.937698 |
| Replayed update, seconds | 0.668554 | 0.663351 |
| Full checkpoint bytes | 1,257,820,683 | 1,257,820,683 |
| Projected time for 4M positive tokens | 69 min 38 s | 34 min 35 s |

Both arms used the same two ordered batches. The CPU role checked both programs on 100 indices each. One eligible first-divergence label had loss 1.170349 and parameter-gradient L2 norm 27.417146. Strict training supplied 2,671 eligible rows from 1,289 families at 28 token positions. The HIP lease lasted 87.513490 seconds. Eighteen readable samples covered 17.000037 of the measured 17.850461 role seconds and indicated 3.714955 busy seconds during that coverage. Peak device allocation was 4,852,388,352 bytes.

The registered estimate uses the slower complete update per token within each arm. Together the two 4M training arms project to 6,252.966991 seconds, or 1 h 44 min 13 s. That leaves 6 h 15 min 47 s in the proposed eight-hour shift, including the four-hour evaluation and replication reservation. The similar second-update timings and the two-update sample limit any speed comparison between objectives. Fresh family supply, full-portfolio evaluation time, prospective qualification and any earned second seed are separate measurements. The audit and rehearsal support continuing with the matched generator.

| Receipt | Identity |
| --- | --- |
| Original producer revision | `46594ef8ba9e4098b1f5759b28635a4537d6d399` |
| Producer source | `sha256:3b1fcf4671f20f8bec24776509c74834f868eb83ced743d511125a9360691cb8` |
| Receipt verifier revision | `48529c5e4b52aa34183bde475428a4dce95e2e5c` |
| Returned ZIP bytes | `sha256:d05a80252197ac54be9c607b1df1d4b17beaeca7927c9a6886b193f27852fe3f` |
| Verified rehearsal aggregate | `sha256:92d1ae2220ebbd2140376a5611269eb740e6eedc0635cd7de49598ba85fe5102` |
| Verified historical audit | `sha256:dea924834ecab7714805bdb4bf08b4c1daf86d994b8b2cc96d55418070b8142c` |
