# Cleanup review records

A reviewer should be able to answer three questions from a proposed change. What disappears, what still consumes it, and how could another maintainer reproduce or reverse the decision? A table in the implementation PR or its review notes is sufficient. Store private source evidence in the operator’s workspace and refer to it by its existing identity.

## Records and relationships

| Record | Fields | Relationship and validation |
| --- | --- | --- |
| Candidate | Baseline revision, tracked path, optional qualified symbol, former purpose, affected group | A method can be a candidate while its class, module and public callers stay supported. A directory name does not stand in for its members. |
| Consumer evidence | Candidate, consumer, way it is reached, exact reference, observation and limits | A consumer may call, name, enumerate, package, authenticate, display or document the candidate. Distinguish a current reader from a writer of the same basename, a citation of an immutable revision, and a match confined to the proposed group. Record the searches that found no additional consumers and the places those searches cannot observe. |
| Disposition | Candidate, decision, rationale, surviving source or replacement if relevant, blocker or reconsideration condition | **Remove** requires evidence that supported and required future uses survive. **Retain** names a consumer. **Deprecate** names the obsolete portion, the live dependency at risk and when withdrawal can be revisited. **Hold** names a missing fact or check. Place any deprecation note outside authenticated bytes when editing them changes a binding. |
| Change group | Common purpose, member paths/symbols, independent commit and parent, dependency on other groups, net files/bytes or symbols, reversal | Group exclusive references, assets and tests with their owner. Split members with different consumers or qualification costs. A tested reversal restores the affected interface at its original external path. |
| Verification evidence | Group, starting revision, comparison revision, environment, input/fixture identity, invocation, expected and observed outcome, comparison with baseline | Record fixtures, installed CPU/native, real-device and private original runs distinctly. Include skips and missing inputs. A known baseline failure remains visible. A group can advance when its own applicable compatibility checks pass. |

A candidate starts under investigation. A traced consumer produces a retention decision or identifies the boundary for deprecation. A candidate with evidence for removal goes through a disposable trial, then its baseline comparison. A failed comparison returns it to investigation. After a successful trial the implementation commit and verification evidence close the record. Reopen the record if a new consumer appears.

## Identities the records must distinguish

| Identity | What it binds | Comparison rule |
| --- | --- | --- |
| Git revision, path and blob | The exact retrieval location and bytes of an old helper or receipt. | Resolve the original commit and path. The round027 continuation also checks the training checkout HEAD itself. |
| Qualification source and installed resources | A named map of inputs and hashes, including virtual package paths, captured at build time. | Compare keys and values, preserving additions, removals and changed bytes. Compare a directly built wheel with a source-distribution rebuild and with the relevant installed command. |
| Image and native identities | Actual container image IDs, native tool/source inputs, selected device, original qualification receipts. | Record each from the environment where it was measured. A matching Python source digest cannot establish either image identity or HIP readiness. |
| Campaign and preparation proof | Accepted campaign ID, purpose and track, frozen graph, runtime, proof version, donors, original input/producer references and caps. | Recover using that campaign’s checkout and original bindings. New source gets its own applicable qualification and separately justified donor compatibility. |
| Task, checkpoint and charge | Task ID, owner/lease, status, original outputs and full-state checkpoint references, journal prefix, reservation/commit key and resource vector. | Preserve completed work and its original charges exactly once. For round027 the authorized new publication has its own key. Repeated recovery must not create another completion or debit. |
| Report and claim | Terminal invocation, bundle identity, public evidence references, verified status, missing evidence and scientific decision. | Claims are scoped to original producers. A bundle can verify while the scientific comparison is incomplete. Private evaluation inputs stay on their permitted side of the boundary. |

## Example review entry

| Candidate | Evidence | Current disposition | Before integration |
| --- | --- | --- | --- |
| `reports/real_data_e2e_report.md` at `a2618bbd` | Legacy checked-in output. The integration test writes a fresh file of that name under its temporary directory. The observed source map and Docker context exclude the tracked path. | Propose removal of the report alone. Its producer, proof-label reader and controls have current uses. | Check current documentation and byte/hash consumers, compare the reporting tests and neighboring paths, and attach the group’s diff and reversal. |

[Measured candidate evidence](research.md) · [Review and operator acceptance contract](contracts/cleanup-acceptance.md)
