# External-source suitability decisions

Research review date: 2026-11-09. These are analysis records, not acquisition or
permission receipts. No live source bytes have been adopted. Runtime snapshots,
revision/hash, access/use approval, group roster and verification budget remain
mandatory freeze inputs. Human choice of a transformation is external_assisted.

| Source | Researched facts and primary references | Proposed semantics and independent checks | Decision and unresolved inputs |
|---|---|---|---|
| FindStat/Sage | [FindStat](https://findstat.org/) describes statistics and maps; its [Sage interface](https://doc.sagemath.org/html/en/reference/databases/sage/databases/findstat.html) supports object/value and distribution queries. The [permutation collection](https://findstat.org/CollectionsDatabase/Permutations/) describes stored sizes through seven. | Reviewed local allowlisted definitions, complete permutation collections n=0..8, exact first/second moment totals; factorial cardinalities, small examples, alternate enumeration and independent total identities. Local enumeration of size eight is a new computation, not a claim that it was downloaded. Related maps and distribution-equivalent statistics stay grouped. | Candidate only for combinatorial-short/v1. Freeze precise statistic IDs/revision, approved use conditions and source relations; cap 46,234 objects and work/bytes. Four independent groups in each role with two tasks/group are required; insufficient support is SOURCE_DEFICIT. Sage is optional producer tooling, never a host/learner dependency. |
| LMFDB/Cremona | [Elliptic curves over Q](https://www.lmfdb.org/EllipticCurve/Q/) exposes curve/isogeny-class labels, integer invariants and rational/real quantities such as j-invariants and regulators. Its source, reliability, completeness and license pages are linked from the database. | Select an exact invariant and mathematical index before designing a task. Curve labels and arbitrary listing order are not sequence indices; isogeny/related-object classes must stay grouped. Verify finite coefficients with independent arithmetic on frozen curve equations. Bound number of objects, coefficient count, work and bytes. Approximate zero locations, real regulators and algebraic objects are not exact integer continuation targets. | NEW_PROFILE_REQUIRED. Immutable download/hash, field semantics, access/use conditions and independent verifier/cost remain unresolved. No bulk ingestion or numeric rounding. Finite polynomial coefficients cannot be zero-padded into an infinite sequence. |
| KnotInfo/Dartmouth | [KnotInfo downloads](https://knotinfo.org/homelinks/database_download.php) provides invariant spreadsheets, references and several polynomial invariant formats; the [project description](https://knotinfo.org/linkinfo/homelinks/about.html) requests citation. The historical Dartmouth URL could not be retrieved in this review. | Knot IDs, mirrors, orientation and invariant conventions require explicit relation groups. Polynomial coefficients have finite support and may use distinct coefficient rings; approximate geometric invariants need a different profile. Independently recompute small approved invariants from fixed diagrams, with crossing/object/work caps before scaling. | NEW_PROFILE_REQUIRED. Freeze original snapshot/hash, exact field/ring conventions, applicable access/use terms, object relations and verified adapter cost. Citation availability alone does not settle redistribution/training permission. |

Decision producer: `progressive.sources.suitability`. It records researched facts
and their URLs separately from unresolved runtime inputs, always with
`executable: false`. A new task profile and matched utility pilot are required for
additional ingestion; source accuracy cannot establish primary scaling success.

For each future decision fill: selected transformation; exact indexing and offset;
finite support; source/group/duplicate relations; immutable bytes/revision/hash;
approved access/use conditions; independent verifier; bounded object/work/byte
reservation; with/without-source arms and paired initializer; deficit or utility
verdict. No source-provided program may execute automatically.
