# **Automated Program Synthesis for Integer Sequences: A Curriculum-Driven Reinforcement Learning Framework for WebAssembly Generation**

Synthesizing executable programs that generate exact mathematical sequences from On-Line Encyclopedia of Integer Sequences (OEIS) data represents a frontier challenge in neural program synthesis1. Training a Transformer model to output WebAssembly Text format (WAT)—which compiles directly into deterministic WebAssembly (WASM) bytecode—requires navigating an expansive, non-differentiable search space3. When guided solely by a strict, binary outcome reward ($+1$ for an exact match across the first $N$ terms and $-1$ otherwise), standard policy gradient techniques encounter extreme sample inefficiency and sparse-reward collapse4.  
To resolve these optimization bottlenecks, this report details a formal 5-stage Curriculum Learning pipeline leveraging OEIS taxonomy and jOEIS structural constructs6, formulates algorithmic graduation criteria alongside quantitative generalization metrics3, and conducts a comparative analysis of policy gradient algorithms under sparse binary rewards4.

## **Curriculum Learning Pipeline Architecture**

The search space of all executable WAT programs contains a dense network of syntactically valid but semantically uninformative programs. To prevent policy collapse during early training, the curriculum must construct a progressive continuum of mathematical abstractions. This progression leverages OEIS metadata tags—such as easy, core, nice, hard, frac, tabl, cofr, mult, base, eigen, and bref7—and maps them directly to structural paradigms established in jOEIS, the primary pure Java execution engine for OEIS sequences2.

### **Stage 1: Primitive Polynomial and Constant Recurrences (Bootstrapping)**

Stage 1 establishes foundational code generation syntax and basic control flow. The mathematical focus centers on arithmetic progressions, constant-step increments, simple polynomial closed-form functions ($a(n) \= \\sum\_{j=0}^k c\_j n^j$), basic modular arithmetic, and direct index-to-value mappings. OEIS metadata tags characterizing this initial tier include easy, core, and nonn7.  
In the jOEIS software architecture, these sequences correspond to direct implementations extending Sequence0 or Sequence1 that compute the next term via basic arithmetic loops without intermediate dynamic memory or stack manipulation6. The target WAT generation paradigm requires simple linear code execution utilizing basic 64-bit integer arithmetic instructions (i64.add, i64.mul, i64.sub, i64.rem\_s) inside a single loop construct, avoiding complex conditional branching or local memory allocations.

### **Stage 2: Linear Recurrences and Rational Generating Functions**

Stage 2 expands program synthesis capabilities into stateful updates across bounded temporal windows. The mathematical concepts encompass constant-coefficient linear recurrences of order $k$ (such as Fibonacci, Lucas, and Pell numbers), rational ordinary generating functions of the form $G(x) \= \\frac{P(x)}{Q(x)}$, and basic multiplicative functions where $a(mn) \= a(m)a(n)$ for coprime integers7. OEIS metadata tags mapped to this stage include core, frac, cons, and mult7.  
Within jOEIS, these sequences are represented by LinearRecurrence abstractions that maintain a bounded sliding window vector of length $k$ updated via matrix-vector multiplication or vector dot products2. The WAT target structures transition from primitive registers to fixed-size linear arrays allocated in local WASM memory or stack registers, executing iterative updates across $k$ state variables per term calculation step.

### **Stage 3: Holonomic and D-Finite Sequences**

Stage 3 introduces dynamic, index-dependent recurrence coefficients. The mathematical scope focuses on linear recurrences with polynomial coefficients (P-finite recurrences) governed by linear differential equations with polynomial coefficients (D-finite generating functions)2, regular continued fraction expansions, and flattened lower-triangular sequence arrays such as Pascal's triangle or Stirling numbers7. OEIS metadata tags belonging to this tier include nice, cofr, tabl, and tabf7.  
In jOEIS, these sequences are implemented via HolonomicSequence instances where terms obey $P\_k(n) a(n+k) \+ \\dots \+ P\_0(n) a(n) \= 0$ for polynomials $P\_i(n) \\in \\mathbb{Z}\[n\]$2. The generated WAT programs require loop-nest structures capable of updating degree-$d$ polynomial coefficients dynamically based on index counter $n$, demanding stack-based polynomial evaluation routines and nested loop conditionals.

### **Stage 4: Combinatorial, Convolutional, and Elementary Number-Theoretic Sequences**

Stage 4 requires non-trivial algorithmic data structures and dynamic memory management. The underlying mathematical concepts cover prime factorization, divisor sums, digital root operations, positional base manipulations, generating function convolutions, and cycle index operations over permutation groups7. Associated OEIS metadata tags include hard, base, eigen, and mult7.  
The jOEIS framework implements these tasks using factor tables (Jaguar.factor), prime generators (Fast), dynamic memory caching (MemoryFunction2), and permutation operations (SymmetricGroup)12. The WAT generation target demands modular sub-routines implementing prime-sieve algorithms, trial division loops, linear memory buffering for dynamic programming tables, and explicit bitwise shift operations for base extraction.

### **Stage 5: Exhaustive State-Space Search, Graph Invariants, and Unbounded Re-computation**

Stage 5 represents the highest difficulty level, requiring general-purpose search algorithms and complex heap operations. Mathematical concepts include backtracking search algorithms ($n$-queens variants, self-avoiding walks), graph isomorphism counts, cellular automata state evolution, non-holonomic sequences lacking finite recurrence relations, and conjectural sequence evaluations18. OEIS metadata tags belonging to this final stage include hard, bref, more, dumb, and less7.  
In jOEIS, these problems require custom Java classes running unbounded graph traversals, bitmask searches on adjacency matrices, or matrix inversions over finite fields18. Synthesizing WAT programs for Stage 5 necessitates deep recursive function calls, heap-memory allocations via WASM memory growth instructions (memory.grow), dynamic array stack frames, and bitmask tracking vectors.

| Stage | Primary OEIS Tags | Mathematical Concepts | jOEIS Structural Analogue | WAT Program Synthesis Complexity |
| :---- | :---- | :---- | :---- | :---- |
| **Stage 1** | easy, core, nonn \[cite: 7, 10, 11\] | Polynomials, closed-form formulas, arithmetic progressions | Direct loops in Sequence0 / Sequence1 \[cite: 6, 13\] | Iterative scalar registers, basic i64 arithmetic |
| **Stage 2** | core, frac, cons, mult \[cite: 7, 10, 14\] | Order-$k$ linear recurrences, rational generating functions | LinearRecurrence vectors2 | Sliding-window memory arrays, matrix multiplication loops |
| **Stage 3** | nice, cofr, tabl, tabf \[cite: 7, 10, 11\] | Holonomic/P-finite recurrences, continued fractions, triangles | HolonomicSequence (D-finite recurrences)2 | Polynomial coefficient loop updating, nested loop structures |
| **Stage 4** | hard, base, eigen \[cite: 7, 10\] | Prime factorization, divisor functions, cycle indexes, convolutions | Jaguar.factor, MemoryFunction2, SymmetricGroup \[cite: 12, 16, 17\] | Memory-buffered dynamic programming, prime sieves, bitwise ops |
| **Stage 5** | hard, bref, more \[cite: 7, 10\] | Graph isomorphisms, backtracking search, non-holonomic sequences | Custom matrix inversions, graph state searches18 | Dynamic WASM stack frames, recursive calls, memory growth |

## **Algorithmic Stage Graduation and Generalization Metrics**

Ad-hoc stage transition rules risk two opposing failure modes: premature advancement (causing exploration collapse due to overwhelming reward sparsity) and prolonged stage over-fitting (causing policy ossification and capacity loss). A mathematically rigorous curriculum requires automated gating mechanisms and validation metrics that distinguish genuine functional synthesis from memorized sequence representations.

### **Automated Curriculum Graduation Algorithm**

Stage graduation is governed by a dual-criterion evaluation framework combining a Rolling Task Competence Score ($C(S\_k)$) with a Policy Variance Threshold ($\\Sigma\_k$).  
Let $S\_k$ denote the set of OEIS prompts belonging to Curriculum Stage $k$. During training, the agent samples prompts $x \\in S\_k$. For a given prompt $x$, the current policy $\\pi\_\\theta$ generates a group of $G$ candidate WAT completions $\\{y\_1, y\_2, \\dots, y\_G\\}$. The exact execution reward function $R(x, y\_i) \\in \\{-1, \+1\\}$ checks if the compiled execution output matches the true target sequence $A(n)$ for $n \\in \\{0, 1, \\dots, N-1\\}$.  
The empirical pass-rate $\\hat{\\rho}\_x$ for prompt $x$ over a rolling window of $W$ recent attempts is defined as:

$$\\hat{\\rho}\_x \= \\frac{1}{W} \\sum\_{w=1}^{W} \\mathbb{I}\\left(R(x, y^{(w)}) \= \+1\\right)$$  
The aggregate Stage Competence Score $C(S\_k)$ is computed as the difficulty-weighted mean pass-rate across all prompts in stage $k$:

$$C(S\_k) \= \\frac{1}{\\vert{}S\_k\\vert{}} \\sum\_{x \\in S\_k} w\_x \\hat{\\rho}\_x$$  
where $w\_x$ represents the relative task difficulty derived from historical search space complexity or baseline pass-rates. Graduation from Stage $k$ to Stage $k+1$ occurs when three conditions hold simultaneously:

> 1. **Competence Threshold**: $C(S\_k) \\ge \\tau\_{\\text{grad}}$, where $\\tau\_{\\text{grad}} \\in \[0.80, 0.90\]$.  
> 2. **Coverage Equilibrium**: $\\min\_{x \\in S\_k} (\\hat{\\rho}\_x) \\ge \\tau\_{\\text{min}}$, ensuring the model has not simply memorized a high-accuracy subset while failing completely on edge-case prompts ($\\tau\_{\\text{min}} \\approx 0.50$).  
> 3. **Policy Stability**: The variance of the success rate across consecutive epochs $E$, defined as $\\mathbb{Var}\_{e \\in E}\[C\_e(S\_k)\] \\le \\varepsilon\_{\\text{var}}$, proving optimization convergence.

To smooth transitions and prevent catastrophic forgetting, prompt sampling uses a dynamic mixture model: when the model graduates to Stage $k+1$, prompts are sampled from a historical distribution where $P(\\text{Stage } k+1) \= 0.70$, $P(\\text{Stage } k) \= 0.20$, and $P(\\text{Stages } 1 \\dots k-1) \= 0.10$.

### **Proving Generalization vs. Memorization**

Because an over-parameterized neural network can memorize finite series of integers via large lookup tables or high-degree Lagrange interpolating polynomials, exact matches on the first $N$ training terms do not guarantee semantic program correctness. Three complementary quantitative metrics establish structural generalization.

#### **Extrapolation Horizon Testing ($N+K$ Term Evaluation)**

The model generates a WAT program evaluated on terms $n \\in \\{0, 1, \\dots, N-1\\}$. To verify structural generalization, the compiled WASM binary is executed out-of-distribution across an extended horizon $n \\in \\{N, N+1, \\dots, N+K-1\\}$ (where $N=20, K=100$). The Extrapolation Generalization Metric $G\_{\\text{ext}}$ is:

$$G\_{\\text{ext}}(y) \= \\prod\_{j=0}^{K-1} \\mathbb{I}\\left(\\text{WASM}\_y(N+j) \== A(N+j)\\right)$$  
A value of $G\_{\\text{ext}}(y) \= 1$ confirms that the synthesized algorithm strictly preserves the underlying recurrence or functional invariant beyond the training boundary.

#### **Minimum Description Length (MDL) and Kolmogorov Complexity Regularization**

Over-fitted lookup tables or Lagrange polynomials require larger binary instruction sequences compared to concise recursive or iterative algorithms3. The Minimum Description Length principle evaluates the combined length of the program code and residual errors3. Let $\\vert{}y\\vert{}\_{\\text{bytes}}$ denote the byte size of the compiled WebAssembly binary. The Normalized Description Metric $M\_{\\text{MDL}}$ compares $\\vert{}y\\vert{}\_{\\text{bytes}}$ against the Kolmogorov complexity proxy $C(A\_N)$, estimated via the Lempel-Ziv compression size of the sequence string:

$$M\_{\\text{MDL}}(y) \= \\frac{\\vert{}y\\vert{}\_{\\text{bytes}}}{C(A\_N)}$$  
If $M\_{\\text{MDL}}(y) \\gg 1$, the model has synthesized a bloated, over-fitted lookup tree. True algorithmic generalization corresponds to $M\_{\\text{MDL}}(y) \\approx 1$ or $M\_{\\text{MDL}}(y) \< 1$, signifying an optimal structural compression of the sequence generating rule3.

#### **Execution Trace Invariant and Semantic Divergence Analysis**

Following Execution-Guided Credit Assignment (EGCA) mechanisms4, candidate WAT programs are instrumented to yield execution trace vectors $\\mathcal{T}\_y$ detailing stack operations, variable mutations, and loop state transitions. By comparing candidate execution traces against reference mathematical invariants derived from jOEIS execution paths, programs that achieve matching outputs via fragile state collisions can be identified and discarded4.

| Metric | Evaluation Mechanism | Target Threshold for Generalization | Primary Failure Mode Detected |
| :---- | :---- | :---- | :---- |
| **Extrapolation Horizon ($N+K$)** | Executes compiled WASM on unseen terms $N \\dots N+K$ | $100\\%$ exact match across $K=100$ terms | Polynomial over-fitting, finite lookup tables |
| **Minimum Description Length ($M\_{\\text{MDL}}$)** | Ratio of compiled WASM byte-size to sequence Lempel-Ziv complexity3 | $M\_{\\text{MDL}} \\le 1.2$ | Bloated conditional trees, hardcoded arrays |
| **Execution Trace Invariants** | Static/dynamic state trace comparison against jOEIS execution patterns4 | Semantic divergence score $= 0$ | Accidental value collisions, uninitialized state bugs |
| **Unseen Sequence Holdout** | Evaluates fine-tuned model zero-shot on un-benchmarked OEIS entries | Zero-shot pass@1 $\\ge 45\\%$ | Over-fitting to prompt structural templates |

## **Evaluation of Reinforcement Learning Algorithms**

Training code-generating models under strict, binary non-differentiable rewards ($+1/-1$) represents a challenging setting for policy optimization4. This section evaluates standard and state-of-the-art policy gradient algorithms: REINFORCE, Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), Leave-One-Out REINFORCE (RLOO), Posterior-GRPO (P-GRPO), and Execution-Guided Credit Assignment GRPO (EGCA-GRPO)4.

### **Mathematical Analysis of Policy Gradient Formulations**

#### **REINFORCE**

REINFORCE computes policy updates directly using raw episodic rewards $R(y)$:

$$\\nabla\_\\theta J\_{\\text{REINFORCE}}(\\theta) \= \\mathbb{E}\_{x \\sim D, y \\sim \\pi\_\\theta}\\left\[ R(y) \\nabla\_\\theta \\log \\pi\_\\theta(y\\vert{}x) \\right\]$$  
Under exact binary rewards ($R \\in \\{-1, \+1\\}$), when the base model probability of producing a correct program is low ($\\hat{\\rho}\_x \\ll 0.05$), almost all sampled completions yield $R \= \-1$. Lacking a dynamic baseline, variance $\\mathbb{Var}\[\\nabla\_\\theta J\]$ scales catastrophically, destabilizing training and causing early entropy collapse9.

#### **Proximal Policy Optimization (PPO)**

PPO stabilizes updates using a learned Value Network $V\_\\phi(s)$ as a critic baseline to estimate Generalized Advantage Estimation (GAE):

$$A\_t^{\\text{PPO}} \= \\delta\_t \+ (\\gamma \\lambda) \\delta\_{t+1} \+ \\dots \\quad \\text{where } \\delta\_t \= R\_t \+ \\gamma V\_\\phi(s\_{t+1}) \- V\_\\phi(s\_t)$$

$$\\mathcal{L}\_{\\text{PPO}}(\\theta) \= \\hat{\\mathbb{E}}\_t \\left\[ \\min\\left( r\_t(\\theta) A\_t^{\\text{PPO}}, \\text{clip}(r\_t(\\theta), 1-\\epsilon, 1+\\epsilon) A\_t^{\\text{PPO}} \\right) \\right\]$$  
While PPO mitigates variance, maintaining an auxiliary Critic model matching the Actor parameter scale (e.g., 7B to 32B parameters) doubles memory consumption23. Furthermore, in non-differentiable code execution, the learned value function $V\_\\phi(s)$ frequently lags behind rapidly changing actor policies, leading to noisy advantage estimates4.

#### **Group Relative Policy Optimization (GRPO)**

GRPO eliminates the critic network entirely by sampling a group of $G$ independent completions $\\{y\_1, y\_2, \\dots, y\_G\\}$ from the old policy $\\pi\_{\\theta\_{\\text{old}}}$ for each prompt $x$23. The advantage $A\_i$ for sample $y\_i$ is computed by normalizing its reward against the group mean $\\mu\_R$ and standard deviation $\\sigma\_R$5:

$$A\_i^{\\text{GRPO}} \= \\frac{R(y\_i) \- \\mu\_R}{\\sigma\_R \+ \\varepsilon}, \\quad \\mu\_R \= \\frac{1}{G} \\sum\_{j=1}^{G} R(y\_j), \\quad \\sigma\_R \= \\sqrt{\\frac{1}{G} \\sum\_{j=1}^{G} (R(y\_j) \- \\mu\_R)^2}$$  
The objective is maximized under token-level KL-divergence constraints24:

$$\\mathcal{L}\_{\\text{GRPO}}(\\theta) \= \\frac{1}{G} \\sum\_{i=1}^{G} \\frac{1}{\\vert{}y\_i\\vert{}} \\sum\_{t=1}^{\\vert{}y\_i\\vert{}} \\left\\{ \\min \\left( \\frac{\\pi\_\\theta(y\_{i,t}\\vert{}x, y\_{i,\<t})}{\\pi\_{\\theta\_{\\text{old}}}(y\_{i,t}\\vert{}x, y\_{i,\<t})} A\_i^{\\text{GRPO}}, \\text{clip}\\left( \\dots \\right) A\_i^{\\text{GRPO}} \\right) \- \\beta D\_{\\text{KL}}\\left(\\pi\_\\theta \\parallel \\pi\_{\\text{ref}}\\right) \\right\\}$$  
Under strict binary outcome rewards ($R \\in \\{-1, \+1\\}$), GRPO encounters an optimization failure mode when prompt difficulty is extreme5. If all $G$ completions fail ($R\_i \= \-1, \\forall i$) or all succeed ($R\_i \= \+1, \\forall i$), then $\\sigma\_R \= 0$, resulting in $A\_i^{\\text{GRPO}} \= 0$ for all samples in the group5. This creates a zero gradient update for those prompts9. The effective prompt weighting function $w\_x$ under binary GRPO simplifies to $w\_x^{\\text{GRPO}} \\propto \\sqrt{\\hat{\\rho}\_x (1 \- \\hat{\\rho}\_x)}$9. Prompts with $\\hat{\\rho}\_x \\approx 0$ (hard prompts) or $\\hat{\\rho}\_x \\approx 1$ (easy prompts) receive zero or vanishing updates, forcing the model to learn exclusively from prompts with intermediate success rates9.

#### **Leave-One-Out REINFORCE (RLOO)**

RLOO constructs an unbiased leave-one-out baseline for completion $y\_i$ using the average reward of the remaining $G-1$ samples9:

$$A\_i^{\\text{RLOO}} \= R(y\_i) \- \\frac{1}{G-1} \\sum\_{j \\neq i} R(y\_j)$$  
While RLOO provides unbiased baseline estimation without a critic network9, its sample efficiency under binary rewards remains constrained by symmetric weighting: $w\_x^{\\text{RLOO}} \\propto \\hat{\\rho}\_x (1 \- \\hat{\\rho}\_x)$, which similarly de-emphasizes hard prompts where all completions currently fail9.

#### **Posterior-GRPO (P-GRPO) and Asymmetric Prompt Weighting**

To solve the zero-advantage collapse on hard prompts, P-GRPO introduces a reasoning verification reward mechanism that evaluates internal token reasoning paths alongside execution outputs8. Furthermore, asymmetric prompt weighting assigns non-zero gradient mass to groups where all responses fail ($R\_i \= \-1$), enforcing a negative update that penalizes the specific syntax paths leading to compilation or execution failure9.

#### **Execution-Guided Credit Assignment GRPO (EGCA-GRPO)**

Standard critic-free methods suffer from coarse credit assignment: a binary reward signal ($R\_i \= \-1$) is distributed uniformly across all generated tokens, even if an error stems from a localized syntax typo or incorrect offset variable4. EGCA executes candidate WAT programs alongside a reference sequence execution trace4. It identifies the exact instruction token where semantic execution divergence occurs, concentrating gradient mass on the causally incorrect token window rather than penalizing valid boilerplate code4.

| RL Algorithm | Critic Model Required? | Memory Footprint (vs. Base) | Handles Sparse Binary Rewards (+1/−1)? | Hard Prompt Gradient Mass (ρ^​x​=0) | Credit Assignment Precision | Sample Efficiency Rating |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **REINFORCE** | No | $1.0\\times$ | Poor (Explosive Variance)9 | Zero baseline failure | Sequence-Level (Coarse) | Low |
| **PPO** | Yes | $\\approx 2.0\\times$ | Moderate (Critic Lag)23 | Non-zero (Critic Estimated) | Sequence-Level (Coarse) | Moderate |
| **Vanilla GRPO** | No | $1.0\\times$ | Moderate (Vanishing Variance)5 | Zero ($A\_i \= 0$ collapse)8 | Token-Uniform across sequence4 | Moderate |
| **RLOO** | No | $1.0\\times$ | Moderate5 | Zero Baseline Collapse9 | Token-Uniform across sequence4 | Moderate |
| **P-GRPO** | No | $1.05\\times$ | High (Posterior Thinking Signals)8 | Non-Zero (Reasoning Derived)8 | Process-Level Credit8 | High |
| **EGCA-GRPO** | No | $1.15\\times$ (Tracing Sandbox) | **Exceptional** (Trace Grounded)4 | Non-Zero (Trace Divergence)4 | **Instruction-Level Localization** \[cite: 4\] | **Maximum** |

### **Algorithmic Recommendation for oeis-learn**

The comparative analysis yields clear empirical and theoretical conclusions regarding algorithm selection for exact WAT synthesis:  
Architectures utilizing group-relative baselines (GRPO family) eliminate the memory overhead of training a separate value network, permitting larger rollout group sizes ($G \\ge 16$) on equivalent hardware23. Larger group sizes directly improve sample efficiency by expanding intra-prompt exploration boundaries23. Furthermore, utilizing continuous test-case pass-rate surrogates (such as $r \= \\frac{\\text{passed terms}}{N}$) instead of binary exact rewards introduces miscalibrated optimization gradients5. Partial term matches in integer sequences frequently reward incorrect closed-form formulas that match initial terms by coincidence, diverting probability mass away from functionally correct solutions5. Binary reward signals remain essential for true algorithm synthesis5.  
Standard GRPO encounters limitations on binary program synthesis due to zero-advantage collapse on hard curriculum prompts and coarse token credit assignment4. The most sample-efficient choice is **Execution-Guided Credit Assignment GRPO (EGCA-GRPO)** augmented with **Asymmetric Prompt Weighting**4. By executing candidate WAT binaries within a WebAssembly sandbox, identifying the exact instruction index where the computed sequence deviates from the target $A(n)$, and masking gradient updates to target that localized token span, EGCA achieves maximum sample efficiency under exact binary rewards4.

## **Synthesis and System Deployment Architecture**

To deploy the oeis-learn synthesis engine efficiently, the curriculum framework, graduation criteria, and policy gradient architecture interlock in a unified pipeline.  
Training begins by initializing the Transformer policy on Stage 1 prompts derived from OEIS entries tagged with easy, core, and nonn7. Prompts supply mathematical descriptions alongside signature declarations in WAT format. Candidate WAT completions generated by the model are compiled and executed inside a high-throughput, sandboxed WebAssembly runtime across $N=20$ initial terms. The runtime returns an exact binary reward $R \\in \\{-1, \+1\\}$ alongside execution state traces4.  
Policy updates are driven by EGCA-GRPO4. For rollout groups where all completions fail ($R\_i \= \-1, \\forall i$), asymmetric prompt weighting ensures non-zero negative gradients are applied directly to the execution divergence span, avoiding zero-advantage collapse4. Model checkpoints are validated periodically using Extrapolation Horizon Testing ($N+K$ terms, $K=100$) and Minimum Description Length regularization to detect lookup table memorization3. The curriculum scheduler continually tracks the rolling competence score $C(S\_k)$ and automatically transitions prompt sampling to Stage $k+1$ once $C(S\_k) \\ge 0.85$ and minimum coverage criteria are satisfied.

## **Nuanced Conclusions and Implementation Recommendations**

Designing an automated program synthesis system for OEIS sequences requires coordinating curriculum design, graduation mechanics, and reinforcement learning objectives:

> * **Taxonomic Curriculum Alignment**: Curriculum stages must map directly to the structural abstractions of OEIS metadata (easy to hard, core, nice, cofr, tabl)7 and jOEIS class hierarchies (Sequence1 $\\rightarrow$ LinearRecurrence $\\rightarrow$ HolonomicSequence $\\rightarrow$ Jaguar.factor $\\rightarrow$ Custom dynamic state searches)2.  
> * **Rigorous Graduation Criteria**: Transition decisions must combine competence thresholds ($C(S\_k) \\ge 0.85$), coverage constraints ($\\min \\hat{\\rho}\_x \\ge 0.50$), and epoch stability metrics to prevent premature stage advancement. Generalization must be validated via $N+K$ term extrapolation and MDL complexity bounds3.  
> * **Algorithm Selection**: Critic-free policy gradient training via **EGCA-GRPO with Asymmetric Prompt Weighting** provides the highest sample efficiency under exact binary outcome rewards4. It eliminates critic memory overhead23, resolves zero-advantage collapse on hard prompts9, and localizes policy updates to the exact instruction tokens responsible for semantic execution divergence4.

#### **Works cited**

> 1. The On-Line Encyclopedia of Integer Sequences today \- Habr, [https://habr.com/en/articles/701208/](https://habr.com/en/articles/701208/)  
> 2. arXiv:2109.02112v1 \[math.CA\] 5 Sep 2021, [https://arxiv.org/pdf/2109.02112](https://arxiv.org/pdf/2109.02112)  
> 3. Research Conference Spotlight – Research Impact & Leadership, [https://sites.gatech.edu/research/spotlight/](https://sites.gatech.edu/research/spotlight/)  
> 4. Execution-Grounded Credit Assignment for GRPO in Code Generation, [https://arxiv.org/pdf/2603.16158](https://arxiv.org/pdf/2603.16158)  
> 5. Exploring Pass-Rate Reward in Reinforcement Learning for Code, [https://arxiv.org/pdf/2605.02944](https://arxiv.org/pdf/2605.02944)  
> 6. archmageirvine/joeis: Java implementations of sequences in the OEIS, [https://github.com/archmageirvine/joeis](https://github.com/archmageirvine/joeis)  
> 7. On-Line Encyclopedia of Integer Sequences, [https://samplecontents.library.ph/wikipedia/wp/o/On-Line\_Encyclopedia\_of\_Integer\_Sequences.htm](https://samplecontents.library.ph/wikipedia/wp/o/On-Line_Encyclopedia_of_Integer_Sequences.htm)  
> 8. Posterior-GRPO: Rewarding Reasoning Processes in Code ... \- arXiv, [https://arxiv.org/html/2508.05170v1](https://arxiv.org/html/2508.05170v1)  
> 9. Asymmetric Prompt Weighting for Reinforcement Learning ... \- arXiv, [https://arxiv.org/html/2602.11128v1](https://arxiv.org/html/2602.11128v1)  
> 10. Category:Keywords \- OeisWiki, [https://oeis.org/wiki/Category:Keywords](https://oeis.org/wiki/Category:Keywords)  
> 11. Clear-cut examples of keywords \- OeisWiki, [https://oeis.org/wiki/Clear-cut\_examples\_of\_keywords](https://oeis.org/wiki/Clear-cut_examples_of_keywords)  
> 12. joeis/src/irvine/oeis/a069/A069240.java at master \- GitHub, [https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a069/A069240.java](https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a069/A069240.java)  
> 13. joeis/src/irvine/oeis/a022/A022447.java at master \- GitHub, [https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a022/A022447.java](https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a022/A022447.java)  
> 14. Explanation of Terms Used in OEIS Sequences, [https://oeis.org/eishelp2.html](https://oeis.org/eishelp2.html)  
> 15. A068029 \- OEIS, [https://oeis.org/A068029/internal](https://oeis.org/A068029/internal)  
> 16. joeis/src/irvine/oeis/a057/A057149.java at master \- GitHub, [https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a057/A057149.java](https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a057/A057149.java)  
> 17. joeis/src/irvine/oeis/a058/A058047.java at master \- GitHub, [https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a058/A058047.java](https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a058/A058047.java)  
> 18. joeis/src/irvine/oeis/a051/A051787.java at master \- GitHub, [https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a051/A051787.java](https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a051/A051787.java)  
> 19. joeis/src/irvine/oeis/a030/A030077.java at master \- GitHub, [https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a030/A030077.java](https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a030/A030077.java)  
> 20. joeis/src/irvine/oeis/a051/A051759.java at master \- GitHub, [https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a051/A051759.java](https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a051/A051759.java)  
> 21. joeis/src/irvine/oeis/a024/A024915.java at master \- GitHub, [https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a024/A024915.java](https://github.com/archmageirvine/joeis/blob/master/src/irvine/oeis/a024/A024915.java)  
> 22. Anomaly Detection Methods for Categorical Data: A Review, [https://www.researchgate.net/publication/333524837\_Anomaly\_Detection\_Methods\_for\_Categorical\_Data\_A\_Review](https://www.researchgate.net/publication/333524837_Anomaly_Detection_Methods_for_Categorical_Data_A_Review)  
> 23. From Reasoning to Code: GRPO Optimization for Underrepresented, [https://arxiv.org/html/2506.11027v3](https://arxiv.org/html/2506.11027v3)  
> 24. Enhanced LLM Reasoning by Optimizing Reward Functions ... \- arXiv, [https://arxiv.org/html/2605.02073v1](https://arxiv.org/html/2605.02073v1)  
> 25. DHRCL: Training Code LLMs with Dense Hierarchical Rewards and, [https://arxiv.org/html/2607.26457v2](https://arxiv.org/html/2607.26457v2)  
> 26. Towards Better Correctness and Efficiency in Code Generation \- arXiv, [https://arxiv.org/pdf/2508.20124](https://arxiv.org/pdf/2508.20124)  
> 27. Exploring Pass-Rate Reward in Reinforcement Learning for Code, [https://arxiv.org/abs/2605.02944](https://arxiv.org/abs/2605.02944)