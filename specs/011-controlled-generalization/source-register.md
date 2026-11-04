# Primary-source register

Reviewed 2026-10-30. The four reports contain 102 numbered references, including duplicate routes, repositories, biographies, explainers and listings. The [citation inventory](citation-inventory.json) maps each supplied reference to the source considered. [source-register.json](source-register.json) records retrieval URLs, available payload hashes and attached-report identities. Neither retrieval nor a report citation establishes evidential support.

Reading depth is explicit:

- **M**: Relevant methods, definitions or results were read directly. The scope column states what was examined; this does not claim every appendix was read.
- **S**: Primary abstract/introduction and a selected method passage were screened for relevance. These sources do not establish numerical targets or adopted algorithm mandates.
- **B**: Background, abstract-only, documentation, or a related presentation; precise scope recorded below.
- **U**: Primary route unavailable. No conclusion depends on its unseen contents.
- **X** in the citation inventory: Secondary/listing material or an unreviewed peripheral reference excluded from the evidence base.

The web-search connector returned errors, so primary PDFs were retrieved directly from proceedings, author sites and arXiv and their text inspected. Four primary works remained inaccessible after attempted retrieval: Execution-Guided Neural Program Synthesis, the supplied Self… policy-optimization paper, Observational Program Similarity, and Alien Codes and Their Automated and Human Explanations. INT was recovered through its primary arXiv copy. Their absence does not supply evidence against their methods. A mistaken lookup of arXiv:1807.03100 resolved to a different text-to-SQL paper and was discarded, not attributed to the requested Execution-Guided paper.

Newest works are identified as 2026 preprints in the review where used; their presence is not independent replication. The FLINT development documentation explicitly warns that portions are undergoing an LLM-assisted revision; no solver correctness or complexity conclusion rests on that prose.

| ID / primary source | Scope | Reading and disposition |
| --- | --- | --- |
| `nesymres` — [Neural Symbolic Regression that Scales](https://proceedings.mlr.press/v139/biggio21a/biggio21a.pdf) | M | Methods, data construction, benchmark definitions and main results |
| `e2e` — [End-to-end Symbolic Regression with Transformers](https://proceedings.neurips.cc/paper_files/paper/2022/file/42eb37cdbefd7abae0835f4b67548c39-Paper-Conference.pdf) | M | Methods and constant/refinement ablations, including Table 2 |
| `recurrence` — [Deep Symbolic Regression for Recurrent Sequences](https://proceedings.mlr.press/v162/d-ascoli22a/d-ascoli22a.pdf) | M | Synthetic integer/recurrent setup, representation and finite evaluation protocol |
| `oeis` — [Learning Program Synthesis for Integer Sequences from Scratch](https://arxiv.org/pdf/2202.11908) | M | Search/check/learn method, budgets and experiments |
| `alien` — [Alien Coding](https://arxiv.org/pdf/2301.11479) | M | Search/check/learn method, retained objectives and additional-term evaluation with timeouts |
| `dreamcoder` — [DreamCoder: Building interpretable hierarchical knowledge representations with wake-sleep Bayesian program learning](https://www.cs.cornell.edu/~ellisk/documents/dreamcoder_with_supplement.pdf) | M | Wake/sleep method and component-ablation passage; author manuscript, not all supplement pages |
| `restem` — [Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://arxiv.org/pdf/2312.06585) | M | EM algorithm, restart policy, experiment design and per-problem sample cap |
| `codeit` — [CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay](https://raw.githubusercontent.com/mlresearch/v235/main/assets/butt24a/butt24a.pdf) | M | Training provenance, hindsight construction, iterative protocol and component ablations |
| `dbca` — [Measuring Compositional Generalization: A Comprehensive Method on Realistic Data](https://arxiv.org/pdf/1912.09713) | M | Atom/compound divergence definitions, split construction and main evaluation |
| `cogs` — [COGS: A Compositional Generalization Challenge Based on Semantic Interpretation](https://aclanthology.org/2020.emnlp-main.731.pdf) | M | Abstract and benchmark construction/generalization-case table |
| `dataconstrained` — [Scaling Data-Constrained Language Models](https://arxiv.org/pdf/2305.16264) | M | Training/scaling setup and repetition results, including Figure 4 |
| `grokking` — [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/pdf/2201.02177) | M | Main experimental setup and regularization comparisons |
| `omnigrok` — [Omnigrok: Grokking Beyond Algorithmic Data](https://arxiv.org/pdf/2210.01117) | M | Abstract, LU mechanism and teacher-student example; no universal extrapolation |
| `prescriptive` — [Prescriptive Scaling Laws for Data Constrained Training](https://arxiv.org/pdf/2605.01640) | M | Training setup, repetition-penalty model and validation sections 3–5 |
| `diverseprior` — [Improving Neural Symbolic Regression with a Diverse Prior and Post-training Alignment](https://arxiv.org/pdf/2610.17994) | M | Prior/post-training methods, benchmark results and ablations in section 4.4 |
| `star` — [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/pdf/2203.14465) | M | Section 3 algorithm and rationalization; supplemental primary source |
| `deepseekmath` — [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300) | M | Section 4.1 GRPO objective/normalization; supplemental primary source |
| `ppo` — [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347) | M | Section 3 clipped surrogate; supplemental primary source |
| `equivalence` — [Program Synthesis with Equivalence Reduction](https://pages.cs.wisc.edu/~aws/papers/vmcai19.pdf) | M | Equivalence assumptions and distinction from observational equivalence |
| `grammaraligned` — [Grammar-Aligned Decoding](https://arxiv.org/pdf/2405.21047) | M | Abstract and constrained-distribution formulation/examples |
| `constrainedsampling` — [Constrained Sampling for Language Models Should Be Easy](https://arxiv.org/pdf/2506.05754) | M | Abstract and constrained-distribution formulation; no adoption of the sampler |
| `deepcoder` — [DeepCoder: Learning to Write Programs](https://arxiv.org/pdf/1611.01989) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `dsr` — [Deep Symbolic Regression: Recovering Mathematical Expressions from Data via Risk-Seeking Policy Gradients](https://arxiv.org/pdf/1912.04871) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `alphacode` — [Competition-Level Code Generation with AlphaCode](https://arxiv.org/pdf/2203.07814) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `symformer` — [SymFormer: End-to-end symbolic regression using transformer-based architecture](https://arxiv.org/pdf/2205.15764) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `differentiablefixer` — [Neural Program Synthesis with a Differentiable Fixer](https://arxiv.org/pdf/2006.10924) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `planning` — [Transformer-based Planning for Symbolic Regression](https://arxiv.org/pdf/2303.06833) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `leanstar` — [Lean-STaR: Learning to Interleave Thinking and Proving](https://arxiv.org/pdf/2407.10040) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `evolutionary` — [Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI](https://arxiv.org/pdf/2507.14172) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `emrl` — [An Expectation-Maximization Perspective on Reinforcement Learning for LLM Reasoning](https://arxiv.org/pdf/2504.18587) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `filtering` — [Scaling Laws for Data Filtering—Data Curation cannot be Compute Agnostic](https://arxiv.org/pdf/2404.07177) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `controlvariables` — [Scalable Neural Symbolic Regression using Control Variables](https://arxiv.org/pdf/2306.04718) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `hrft` — [HRFT: Mining High-Frequency Risk Factor Collections End-to-End via Transformer](https://arxiv.org/pdf/2408.01271) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `l2gsr` — [L2G-SR: Learning to Guide LLM-based Symbolic Regression](https://arxiv.org/pdf/2610.20685) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `grcrl` — [GrammarRL: Effective Grammar-Constrained Decoding via Reinforcement Learning](https://arxiv.org/pdf/2609.39869) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `groupprioritized` — [RLVR without Ineffective Samples: Group Prioritized Off-Policy Optimization for LLM Reasoning](https://arxiv.org/pdf/2606.01281) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `softsequence` — [Soft Sequence Policy Optimization](https://arxiv.org/pdf/2602.19327) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `mismatch` — [Rethinking Training–Inference Mismatch in LLM Reinforcement Learning](https://arxiv.org/pdf/2609.32444) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `advantageclipped` — [Towards Better Training Signal: Advantage Clipped Policy Optimization](https://arxiv.org/pdf/2609.36816) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `diffusionem` — [Diffusion Alignment as Variational Expectation-Maximization](https://proceedings.iclr.cc/paper_files/paper/2026/file/3469b211b829b39d2b0cfd3b880a869c-Paper-Conference.pdf) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `mining` — [Mining or Synthesis? Rethinking Exploration Efficiency in Iterative Alignment of Mathematical Reasoning](https://aclanthology.org/2026.findings-emnlp.1097.pdf) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `alignment` — [Improving Grammar Constraints Generation Alignment by Sampling Highly Probable Playouts](https://www.lamsade.dauphine.fr/~cazenave/papers/GBFSGS_rework.pdf) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `mdlformer` — [Symbolic regression via MDLformer-guided search: from minimizing prediction error to minimizing description length](https://fi.ee.tsinghua.edu.cn/public/publications/23c6426c-f020-11f0-b382-2aaffa21d846.pdf) | S | Abstract/intro and selected method passage screened; no effect-size or implementation mandate derived |
| `int` — [INT: An Inequality Benchmark for Evaluating Generalization in Theorem Proving](https://arxiv.org/pdf/2007.02924) | B | Abstract/introduction; retrieved primary arXiv copy after OpenReview failed |
| `psychology` — [A Psychology-Driven User Agent for Recommender Systems](https://arxiv.org/pdf/2610.13989) | B | Abstract read; unrelated to the pilot conditioning mechanism; excluded |
| `thesis` — [Learning-Aided Program Synthesis and Verification](https://www.cis.upenn.edu/~mhnaik/theses/xujie_si_thesis.pdf) | B | Thesis abstract read; broad background, not an experimental basis |
| `synthesissurvey` — [Program Synthesis](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/10/program_synthesis_now.pdf) | B | Contents screened; not used as primary empirical evidence |
| `chunking` — [Improving Inductive Program Synthesis with Probabilistic Chunking](https://project-archive.inf.ed.ac.uk/ug4/20233908/ug4_proj.pdf) | B | Abstract screened; language-abstraction proposal outside scope |
| `improvementsabstract` — [Improvements in Program Synthesis for Integer Sequences](https://aitp-conference.org/2023/abstract/AITP_2023_paper_8.pdf) | B | Complete short AITP abstract read; cross-checked against Alien Coding |
| `improvementsslides` — [Improvements in Program Synthesis for Integer Sequences — slides](https://aitp-conference.org/2023/slides/TG.pdf) | B | Retrieved; numerical claims checked in underlying paper/abstract, not relied on independently |
| `flint-hnf` — [FLINT Hermite normal form documentation](https://flintlib.org/doc/fmpz_mat/hnf.html) | B | HNF API contracts inspected; no new solver requirement adopted |
| `flint-algorithms` — [FLINT Hermite and Smith normal forms algorithms](https://flintlib.org/doc/algorithms/linear-algebra/hermite-smith.html) | B | Retrieved/context screened; no complexity or applicability claim adopted |
| `lecture` — [Program Synthesis course, Lecture 3](https://people.csail.mit.edu/asolar/SynthesisCourse/Lecture3.htm) | B | HTML retrieved; background not relied on |
| `int-openreview` — [INT — supplied OpenReview route](https://openreview.net/pdf?id=O6LPudowNQm) | U | 403; primary arXiv copy successfully read as int |
| `executionguided-original` — [Execution-Guided Neural Program Synthesis](https://openreview.net/pdf?id=H1gfOiAqYm) | U | 403; excluded from supporting evidence |
| `selfpolicy` — [Principled Policy Optimization for LLMs via Self… (supplied truncated title)](https://openreview.net/pdf?id=HVciz8hi1c) | U | 403; title lookup did not resolve an authoritative accessible copy; excluded |
| `observational` — [Inductive Program Synthesis Guided by Observational Program Similarity](https://par.nsf.gov/servlets/purl/10497577) | U | 503; excluded; finite-equivalence distinction checked in Smith/Albarghouthi |
| `alienexplanations` — [Alien Codes and Their Automated and Human Explanations](https://grid01.ciirc.cvut.cz/~mptp/cicm25oeis.pdf) | U | 503; excluded; earlier primary OEIS papers used separately |

## Evidence boundary

Adopted decisions and their specific sources are in [research-review.md](research-review.md). The specification's budgets, seed count, coverage defaults, completion floor, replay limits and promotion screen are prospective engineering choices, not measured effects extracted from these papers. Primary research supports hypotheses and design distinctions; current repository code establishes what the implementation actually does.

This review does not reproduce the cited experiments, independently re-audit unavailable pilot archives, or claim a complete reading of every background thesis, appendix and secondary link. Unavailable or merely screened references are not used to fill gaps in the evidence.
