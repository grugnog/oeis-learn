"""Shared checkpoint-to-claim discovery pipeline orchestration."""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from oeis_learn.data.benchmark import load_benchmark_manifest
from oeis_learn.data.models import SequenceRef
from oeis_learn.data.symbolic_definitions import SymbolicDefinitionRegistry
from oeis_learn.discovery.numerical_validator import validate_numerical_relation
from oeis_learn.discovery.pslq_solver import PslqRelationSolver
from oeis_learn.discovery.relation_identity import canonicalize_relation, is_trivial_relation
from oeis_learn.discovery.symbolic_prover import SymbolicProver
from oeis_learn.discovery.vector_search import VectorRelationSearcher
from oeis_learn.evaluation.checkpoint import load_checkpoint_v2

logger = logging.getLogger("oeis_learn.discovery")


def run_discovery_pipeline(
    checkpoint_path: str,
    manifest_path: str = "data/benchmarks/trustworthy_synthesis_v1.json",
    protocol_path: str = "configs/discovery_protocol_v1.json",
    definitions_path: str = "data/benchmarks/symbolic_definitions_v1.json",
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Executes the full latent proposal, numerical validation, and symbolic verification pipeline."""
    dev = device or torch.device("cpu")
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1. Load Protocol
    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol_dict = json.load(f)

    # 2. Load Checkpoint and Registry
    encoder, _, checkpoint_prov = load_checkpoint_v2(checkpoint_path, device=dev)
    registry = SymbolicDefinitionRegistry(definitions_path)
    from oeis_learn.experiments.artifacts import compute_canonical_digest

    if protocol_dict.get(
        "definition_registry_sha256"
    ) != registry.registry_sha256 or protocol_dict.get("protocol_id") != compute_canonical_digest(
        protocol_dict, "protocol_id"
    ):
        raise ValueError("Discovery protocol/registry identity mismatch")
    manifest = load_benchmark_manifest(manifest_path)

    # 3. Extract L2-Normalized Sequence Embeddings
    encoder.eval()
    embeddings: Dict[str, np.ndarray] = {}
    terms_dict: Dict[str, List[int]] = {}

    with torch.no_grad():
        for t in manifest.targets:
            int_terms = [int(x) for x in t.observed_terms] + [int(x) for x in t.unseen_terms]
            terms_dict[t.oeis_id] = int_terms
            z = encoder.forward_from_sequences([int_terms[:20]], device=dev)
            emb = z.mean(dim=1).squeeze(0).cpu().numpy()
            norm = np.linalg.norm(emb)
            norm_emb = emb / norm if norm > 1e-8 else emb
            embeddings[t.oeis_id] = norm_emb

    # 4. Propose Latent Candidates via Nearest-Neighbor Vector Relation Search
    dist_thresh = float(protocol_dict.get("distance_threshold", 0.8))
    max_cands = int(protocol_dict.get("max_candidates", 50))
    searcher = VectorRelationSearcher(eps_distance=dist_thresh, normalize_l2=True)
    raw_candidates = searcher.search_additive_triples(embeddings, max_candidates=max_cands)

    pslq = PslqRelationSolver(precision_digits=int(protocol_dict.get("precision_digits", 500)))
    prover = SymbolicProver()

    claims: List[Dict[str, Any]] = []
    seen_claim_ids: Dict[str, str] = {}
    dispositions: List[Dict[str, Any]] = []

    num_latent = len(raw_candidates)
    num_unique = 0
    num_dup = 0
    num_conj = 0
    num_proven = 0
    num_rejected = 0
    num_insuff = 0

    val_indices = protocol_dict.get("validation_indices", list(range(10)))
    uns_indices = protocol_dict.get("unseen_indices", list(range(20, 120)))
    search_indices = protocol_dict.get("search_indices", list(range(10, 20)))

    for idx, cand in enumerate(raw_candidates):
        cand_id = f"cand_{idx:03d}"
        seq_ids = list(cand.sequences)
        seq_vectors = [terms_dict[sid] for sid in seq_ids if sid in terms_dict]

        if len(seq_vectors) != len(seq_ids):
            dispositions.append(
                {
                    "candidate_id": cand_id,
                    "disposition": "INSUFFICIENT_EVIDENCE",
                    "claim_id": None,
                    "reason_code": "MISSING_SEQUENCE_TERMS",
                }
            )
            num_insuff += 1
            continue

        # Run PSLQ coefficient detection
        rel_vec, conf = pslq.find_relation(seq_vectors, fitting_indices=search_indices)
        if rel_vec is None:
            dispositions.append(
                {
                    "candidate_id": cand_id,
                    "disposition": "INSUFFICIENT_EVIDENCE",
                    "claim_id": None,
                    "reason_code": "NO_INTEGER_RELATION",
                }
            )
            num_insuff += 1
            continue

        operands = [SequenceRef(sid, 1, 0) for sid in seq_ids]
        is_triv, triv_reason = is_trivial_relation(operands, rel_vec)
        if is_triv:
            dispositions.append(
                {
                    "candidate_id": cand_id,
                    "disposition": "REJECTED_TRIVIAL",
                    "claim_id": None,
                    "reason_code": triv_reason,
                }
            )
            num_rejected += 1
            continue

        # Canonicalize relation
        can_rel = canonicalize_relation(operands, rel_vec)
        claim_id = can_rel.claim_id

        # Deduplication
        if claim_id in seen_claim_ids:
            dispositions.append(
                {
                    "candidate_id": cand_id,
                    "disposition": "DUPLICATE_OF",
                    "claim_id": claim_id,
                    "reason_code": f"Equivalent to claim {seen_claim_ids[claim_id]}",
                }
            )
            num_dup += 1
            continue

        seen_claim_ids[claim_id] = cand_id
        num_unique += 1

        # Numerical validation across all validation and unseen terms
        num_evidence = validate_numerical_relation(
            operands=can_rel.operands,
            coefficients=[int(c) for c in can_rel.coefficients],
            sequence_terms_dict=terms_dict,
            validation_indices=val_indices,
            unseen_indices=uns_indices,
            search_indices=search_indices,
        )

        status_history = [
            {
                "status": "LATENT_CANDIDATE",
                "created_at": now_utc,
                "evidence_type": "EMBEDDING_TRIPLE",
            }
        ]
        status = "LATENT_CANDIDATE"
        sym_evidence = None
        rejection = None

        if num_evidence.outcome == "VERIFIED":
            status = "NUMERICALLY_VERIFIED_CONJECTURE"
            status_history.append(
                {
                    "status": "NUMERICALLY_VERIFIED_CONJECTURE",
                    "created_at": now_utc,
                    "evidence_type": "EXACT_PARTITIONED_VALIDATION",
                }
            )
            # Attempt general symbolic proof
            sym_outcome, sym_evidence = prover.prove_canonical_relation(can_rel, registry)
            # Conditional formula identities do not establish OEIS correspondence.
            num_conj += 1
        elif num_evidence.outcome == "COUNTEREXAMPLE":
            status = "REJECTED"
            rejection = {
                "reason": "COUNTEREXAMPLE",
                "first_counterexample": num_evidence.first_counterexample,
            }
            num_rejected += 1
        else:
            status = "INSUFFICIENT_EVIDENCE"
            num_insuff += 1

        dispositions.append(
            {
                "candidate_id": cand_id,
                "disposition": "NEW_CLAIM",
                "claim_id": claim_id,
                "reason_code": None,
            }
        )

        latent_entry = {
            "candidate_id": cand_id,
            "checkpoint_sha256": checkpoint_prov.checkpoint_sha256,
            "embedding_version": "tri_stream_strict_fp32",
            "vector_distance": float(cand.vector_distance),
            "search_parameters": {"distance_threshold": dist_thresh},
            "seed": seed,
            "backend_versions": {"torch": torch.__version__},
        }

        claims.append(
            {
                "schema_version": "1.0",
                "relation": can_rel.to_dict(),
                "status": status,
                "latent_evidence": [latent_entry],
                "numerical_evidence": num_evidence.to_dict(),
                "symbolic_evidence": sym_evidence,
                "rejection": rejection,
                "status_history": status_history,
            }
        )

    report_id = f"rep_disc_{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"
    summary = {
        "latent_candidates": num_latent,
        "unique_claims": num_unique,
        "duplicate_candidates": num_dup,
        "numerical_conjectures": num_conj,
        "symbolically_proven": num_proven,
        "rejected": num_rejected,
        "insufficient_evidence": num_insuff,
    }

    return {
        "schema_version": "1.0",
        "report_id": report_id,
        "run_id": f"discovery_run_{seed}",
        "created_at": now_utc,
        "checkpoint_sha256": checkpoint_prov.checkpoint_sha256,
        "benchmark_manifest_sha256": manifest.manifest_sha256,
        "protocol": protocol_dict,
        "claims": claims,
        "candidate_dispositions": dispositions,
        "summary": summary,
        "errors": [],
        "mode": "legacy_embedding_diagnostic",
        "program_provenance": "not_established",
    }


def run_program_discovery(
    archive,
    *,
    snapshot=None,
    fitting_indices=tuple(range(10)),
    validation_indices=tuple(range(10, 20)),
    final_indices=tuple(range(20, 100)),
    max_candidates=50,
):
    """Strict training-program analysis. External databases/definitions have no argument.

    Every source and output is read through ProgramArchive's verified training
    view, preserving multiple implementations and actual decoder genealogy.
    Coefficients and selection are sealed before independent final validation.
    """
    from oeis_learn.discovery.program_features import ProgramFeatureIndex
    from oeis_learn.discovery.novelty import seal_proposal
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    parts = [tuple(p) for p in (fitting_indices, validation_indices, final_indices)]
    if any(
        not p or len(set(p)) != len(p) or any(type(i) is not int or i < 0 for i in p) for p in parts
    ) or any(set(parts[i]) & set(parts[j]) for i in range(3) for j in range(i)):
        raise ValueError("disjoint nonempty program discovery partitions required")
    snapshot = archive.freeze() if snapshot is None else snapshot
    index = ProgramFeatureIndex(archive, snapshot=snapshot)
    index.validate()
    view = dict(archive.training_view(snapshot))
    vectors = {}
    terms = {}
    missing = []
    for row in index.rows:
        key = row["key"]
        outputs = view[key]["entry"]["outputs"]
        if any(i >= len(outputs) for i in sum(parts, ())):
            missing.append({"program": key, "status": "INSUFFICIENT_EVIDENCE"})
            continue
        terms[key] = [int(v) for v in outputs]
        # Only fitting values participate in retrieval. Avoid converting huge
        # authoritative integers to float without a disclosed bounded feature.
        scale = max(1, max(abs(terms[key][i]) for i in parts[0]))
        numerical = [float(terms[key][i] / scale) for i in parts[0]]
        vectors[key] = np.asarray(numerical, dtype=np.float64)
    search = VectorRelationSearcher(neighbors=8, max_pairs=4096)
    candidates = search.search_additive_triples(vectors, max_candidates=max_candidates)
    claims = []
    seen = set()
    fit_solver = PslqRelationSolver()
    for candidate in candidates:
        sources = list(candidate.sequences)
        fit = fit_solver.fit([terms[k] for k in sources], parts[0])
        if fit.status != "FIT":
            continue
        operands = [SequenceRef(k, 1, 0) for k in sources]
        relation = canonicalize_relation(operands, fit.coefficients)
        if relation.claim_id in seen:
            continue
        seen.add(relation.claim_id)
        ordered = [o.oeis_id for o in relation.operands]
        proposal = seal_proposal(
            {
                "version": "program-relation/v1",
                "archive_root_sha256": snapshot["root_sha256"],
                "feature_index_sha256": index.identity,
                "relation": relation.to_dict(),
                "programs": [
                    {
                        "key": key,
                        "source_sha256": view[key]["entry"]["source_sha256"],
                        "codec_sha256": view[key]["entry"]["codec_sha256"],
                        "generation": view[key]["provenance"],
                    }
                    for key in ordered
                ],
                "partitions": parts,
                "fit_rank": fit.rank,
                "fit_nullity": fit.nullity,
                "fit_attempts": fit.attempts,
                "operator_priors": {"integer_linear_arity3": 1},
                "evidence_kind": "empirical_relation",
                "proof_status": "not_claimed",
            }
        )
        evidence = validate_numerical_relation(
            relation.operands,
            relation.coefficients,
            terms,
            search_indices=parts[0],
            validation_indices=parts[1],
            unseen_indices=parts[2],
        )
        claims.append(
            {
                "proposal": proposal,
                "finite_evidence": evidence.to_dict(),
                "status": "FINITE_VALIDATED"
                if evidence.outcome == "VERIFIED"
                else evidence.outcome,
                "novelty": "novelty_unknown",
            }
        )
    from oeis_learn.discovery.transforms import discover_transforms
    transformations=discover_transforms(terms,fit_indices=parts[0],validation_indices=parts[1],final_indices=parts[2])
    report = {
        "version": "program-discovery/v1",
        "archive_root_sha256": snapshot["root_sha256"],
        "claims": claims,
        "unavailable": missing,
        "retrieval": search.last_metrics,
        "proposals": len(candidates),
        "information_policy": "verified_training_programs_only",
        "external_lookup_performed": False,
        "transforms": transformations,
    }
    report["report_sha256"] = digest(report)
    return report
