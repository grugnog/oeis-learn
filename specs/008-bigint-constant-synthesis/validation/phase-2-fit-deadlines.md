# Focused Phase 2 correction: bounded fitting completion

The user authorized the focused correction after the reviewed replay hit its
two-second worker deadline without returning a result. This reopens the
T032–T036/T058 fitting and containment boundary under Spec 008. It does not
resume or replace the completed negative T096 attempt.

## Fixed scope

Keep the two-second native/production ceiling, independent-reference reservation,
four-second candidate call and sixty-second portfolio. Native work uses the
earlier of the caller deadline and two seconds from entry. Freeze internal
policy `native-fit-deadlines/v1`: reserve5% of the remaining native allowance
for response delivery and25% for final production checks. Search and symbolic
work share the preceding70%. These proportional reserves shrink with a shorter
caller allowance and never extend or restart it. Expired work remains expired.
External supervision remains the hard bound under scheduler/solver overrun.

Preserve the language, abstraction, solver subset/bounds/seeds, maximum eight
assignments, discovered-assignment ranking, production checks and independent
admission. An early SAT assignment may be checked after enumeration runs out of
its search slice; UNKNOWN itself never supplies a model. Incomplete uniqueness
and search outcomes remain explicit. Eight assignments already prove multiple
solutions, so no ninth solver check is needed to establish ambiguity. A
single retained assignment still needs the blocking check to establish unique
versus multiple and cannot be called unique after a timeout.

No raw fallback, threshold change, hidden-value use or training experiment is
included. Binary/source/cache identities change; future qualification must bind
the new image. This correction improves bounded delivery and use of discovered
assignments, without promising success for the previously timed-out seed.

## Work and acceptance

- [ ] F001 Add native regressions for deadline nesting, short/expired caller
  allowances, late enumeration with early SAT assignments, production overrun,
  the eight-assignment cap and single-model ambiguity.
- [ ] F002 Implement proportional deadline partitions and route symbolic,
  affine/SMT and production work through their existing parent allowance.
- [ ] F003 Retain bounded search/deadline telemetry without bypassing independent
  admission, and stop the redundant ninth check after a full ambiguous shortlist.
- [ ] F004 Add packaged-native regressions for the observed candidate and
  independent fixtures, normal response delivery and healthy worker recovery.
- [ ] F005 Run available local checks, push incremental commits and publish
  container validation/replay commands. Actual native/HIP evidence is reported
  separately; do not wait for PR CI.
