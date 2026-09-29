# OEIS Corpus Precision Census & Bit-Width Feasibility Report

- **Generated At**: 2026-09-06T17:20:21.008288+00:00
- **Database Evaluated**: `data/oeis_corpus.duckdb`
- **Total Sequences Ingested**: 399,005

## 1. Executive Summary & Architectural Finding

At the standard 100-term extrapolation horizon ($N=100$), a fixed **4 x i64 (256-bit, `i256x4_v1`)** representation achieves:
- **99.51% coverage** across core mathematical invariants.
- **99.56% coverage** across all jOEIS computable sequences.
- **99.51% coverage** across the entire OEIS global corpus.

## 2. Multi-Limb Precision Coverage Matrix (Horizon N=100)

| Cohort Slice | Total Sequences | 1 x i64 (63b) | 2 x i64 (127b) | 4 x i64 (255b) | 8 x i64 (511b) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **All Sequences** | 399,005 | 83.2% | 97.0% | **99.5%** | 100.0% |
| **Core / Foundational (core/nice)** | 279,005 | 80.3% | 97.0% | **99.5%** | 100.0% |
| **jOEIS Computable** | 74,986 | 85.0% | 96.9% | **99.6%** | 100.0% |
| **Closed-Form / Formula** | 49,694 | 76.7% | 94.2% | **98.9%** | 100.0% |
| **Stage 1: Polynomials** | 278,485 | 80.5% | 97.1% | **99.5%** | 100.0% |
| **Stage 2: Linear Recurrences** | 9,136 | 85.7% | 95.0% | **99.1%** | 100.0% |
| **Stage 3: Holonomic / Catalan** | 3,629 | 74.9% | 93.9% | **99.3%** | 100.0% |
| **Stage 4: Primes / Number Theory** | 91,115 | 92.2% | 97.4% | **99.5%** | 100.0% |

## 3. Bit-Width Percentiles Across Progressive Horizons

| Cohort Slice | Horizon | p50 (bits) | p75 (bits) | p90 (bits) | p95 (bits) | p99 (bits) | Max (bits) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| All Sequences | N=20 | 14 | 41 | 77 | 101 | 206 | 947 |
| All Sequences | N=50 | 18 | 50 | 78 | 102 | 207 | 947 |
| All Sequences | N=100 | 19 | 50 | 78 | 102 | 207 | 947 |
| Core / Foundational (core/nice) | N=20 | 17 | 50 | 80 | 103 | 206 | 899 |
| Core / Foundational (core/nice) | N=50 | 24 | 57 | 82 | 104 | 207 | 899 |
| Core / Foundational (core/nice) | N=100 | 24 | 57 | 82 | 104 | 207 | 899 |
| jOEIS Computable | N=20 | 13 | 39 | 74 | 98 | 205 | 697 |
| jOEIS Computable | N=50 | 17 | 47 | 76 | 99 | 207 | 697 |
| jOEIS Computable | N=100 | 17 | 47 | 76 | 99 | 207 | 697 |
| Closed-Form / Formula | N=20 | 17 | 55 | 96 | 139 | 257 | 842 |
| Closed-Form / Formula | N=50 | 25 | 61 | 97 | 140 | 258 | 842 |
| Closed-Form / Formula | N=100 | 25 | 61 | 98 | 140 | 258 | 842 |
| Stage 1: Polynomials | N=20 | 16 | 49 | 80 | 102 | 204 | 899 |
| Stage 1: Polynomials | N=50 | 23 | 57 | 81 | 103 | 206 | 899 |
| Stage 1: Polynomials | N=100 | 24 | 57 | 81 | 103 | 206 | 899 |
| Stage 2: Linear Recurrences | N=20 | 13 | 31 | 77 | 125 | 251 | 675 |
| Stage 2: Linear Recurrences | N=50 | 22 | 42 | 79 | 126 | 251 | 675 |
| Stage 2: Linear Recurrences | N=100 | 22 | 42 | 79 | 126 | 251 | 675 |
| Stage 3: Holonomic / Catalan | N=20 | 20 | 57 | 96 | 144 | 235 | 422 |
| Stage 3: Holonomic / Catalan | N=50 | 33 | 64 | 97 | 144 | 235 | 422 |
| Stage 3: Holonomic / Catalan | N=100 | 33 | 64 | 97 | 144 | 235 | 422 |
| Stage 4: Primes / Number Theory | N=20 | 11 | 19 | 44 | 78 | 204 | 947 |
| Stage 4: Primes / Number Theory | N=50 | 13 | 23 | 53 | 81 | 208 | 947 |
| Stage 4: Primes / Number Theory | N=100 | 14 | 23 | 54 | 81 | 208 | 947 |

## 4. Asymptotic Growth Rate Breakdown

| Cohort Slice | Bounded / Periodic | Polynomial | Moderate Exp | Fast Exp | Factorial | Super Exp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| All Sequences | 49.1% | 34.4% | 13.6% | 2.3% | 0.4% | 0.0% |
| Core / Foundational (core/nice) | 45.1% | 35.4% | 16.5% | 2.4% | 0.4% | 0.0% |
| jOEIS Computable | 50.6% | 34.6% | 11.8% | 2.5% | 0.4% | 0.0% |
| Closed-Form / Formula | 44.1% | 33.4% | 17.1% | 4.5% | 0.9% | 0.0% |
| Stage 1: Polynomials | 45.2% | 35.6% | 16.3% | 2.3% | 0.4% | 0.0% |
| Stage 2: Linear Recurrences | 46.4% | 39.5% | 9.3% | 3.8% | 0.9% | 0.0% |
| Stage 3: Holonomic / Catalan | 41.2% | 34.0% | 18.8% | 5.3% | 0.6% | 0.0% |
| Stage 4: Primes / Number Theory | 63.1% | 29.3% | 4.9% | 1.9% | 0.4% | 0.0% |

## 5. Architectural Recommendation for Neuro-Symbolic Synthesis

1. **Adopt `i256x4_v1` (4 x i64) as the Standard Wide Integer Profile**:
   - Covers >99% of computable, formula-based, and core number theory sequences through 100 terms.
   - Retains pure WebAssembly value-stack semantics: functions return `(result i64 i64 i64 i64)` without requiring linear memory allocation (`malloc`/`free`) or pointer manipulation.
   - Preserves compatibility with exact SMT/Diophantine solvers (Z3 supports native `BitVec(256)` operations with zero translation overhead).
2. **Graceful Handling of Super-Factorial Sequences**:
   - For sequences growing faster than $O(6^n)$ (e.g. $100!$, Bell numbers $B_{100}$), evaluate up to the 256-bit boundary ($n \le 57$ for factorials) rather than forcing heap-allocated dynamic BigInt.
   - Eliminates out-of-memory traps and memory leaks in the Wasmtime execution sandbox.