# **Mitigating Degenerate Constant Collapses in RL-Guided Program Synthesis**

## **1\. Background and Failure Dynamics of Shortcut Collapse**

In neuro-symbolic program synthesis, an autoregressive neural policy $\\pi\_\\theta(P \\mid \\mathcal{S})$—typically instantiated via a Transformer architecture—is tasked with generating executable programmatic bytecode $P$ (such as WebAssembly Text or postfix stack primitives) from a formal task specification $\\mathcal{S}$ consisting of input-output pairs $\\{(n, y\_n)\\}\_{n=0}^{N-1}$1. The goal is to synthesize an algorithmic routine $P$ such that $P(n) \= y\_n$ for all training inputs $n \\in \\{0, 1, \\dots, N-1\\}$, while ensuring the synthesized logic generalizes out-of-distribution (OOD) to arbitrary evaluation inputs $n \\ge N$3.  
Under Reinforcement Learning with Verifiable Rewards (RLVR) frameworks utilizing policy optimization algorithms such as Proximal Policy Optimization (PPO) or Group Relative Policy Optimization (GRPO), the ground-truth environment reward is binary, sparse, and verifiable1:

$$R\_{\\text{exact}}(P, Y) \= \\begin{cases} \+1.0 & \\text{if } P(n) \= y\_n, \\; \\forall n \\in \\{0, 1, \\dots, N-1\\} \\\\ \-1.0 & \\text{otherwise} \\end{cases}$$  
To alleviate the extreme sample inefficiency associated with sparse terminal rewards, dense surrogate reward shaping is commonly introduced, combining compiler validity, prefix matching, and continuous output distance7:

$$R\_{\\text{surr}}(P, Y) \= w\_{\\text{comp}} R\_{\\text{comp}}(P) \+ w\_{\\text{prefix}} R\_{\\text{prefix}}(P, Y) \+ w\_{\\text{dist}} R\_{\\text{dist}}(P, Y)$$  
where $R\_{\\text{comp}}(P) \\in \\{0, 1\\}$ measures syntactic compilation validity, $R\_{\\text{prefix}}(P, Y) \= \\frac{1}{N} \\max \\{k \\mid P(n) \= y\_n, \\forall n \< k\\}$ evaluates sequential prefix correctness, and $R\_{\\text{dist}}(P, Y) \= 1.0 \- \\frac{1}{N} \\sum\_{n=0}^{N-1} \\tanh(\\alpha \\vert{}P(n) \- y\_n\\vert{})$ provides a normalized continuous distance metric7.

### **Mechanics of Constant Shortcut Collapse**

When dynamic grammar masking or constrained decoding is deployed, $R\_{\\text{comp}}(P)$ is artificially locked at $1.0$ because the decoder is strictly prevented from emitting syntactically invalid tokens11. Under these conditions, the policy rapidly uncovers a pathological local optimum: emitting short, static programs that ignore the input parameter $n$ entirely (e.g., returning a fixed constant $C$, such as (module (func (export "compute") (param $n i32) (result i64) i64.const 16)))11.

| Paradigm Component | Standard Sparse RLVR | Unguided Dense Surrogate RL | Constrained Syntax \+ Dense Surrogates |
| :---- | :---- | :---- | :---- |
| **Compiler Reward ($R\_{\\text{comp}}$)** | Binary feedback (+1.0 or \-1.0)6. | Stochastic; frequent penalties for syntax traps7. | Artificially locked at \+1.0 via dynamic grammar masking11. |
| **Dominant Policy Behavior** | High sample inefficiency; slow exploration6. | Trapped in non-compiling syntax states7. | **Degenerate Constant Shortcut Collapse**11. |
| **Entropy of Generated Code** | High initial token entropy11. | Moderate token entropy11. | Collapses to minimum token sequence length11. |
| **Input Dependence $I(n; P(n))$** | High in successful trajectories11. | Low; dominated by syntax errors7. | Collapses strictly to zero ($I(n; P(n)) \= 0$)11. |
| **Extrapolation ($n \\ge N$)** | Perfect if exact match found6. | Zero; execution fails7. | Total failure; outputs constant $C$ for all $n$11. |

This state represents a dominant local attractor in the policy gradient landscape due to three compounding mechanisms:

> 1. **Entropy Minimization & Policy Stability:** Generating a short constant payload requires minimal autoregressive sequence length, minimizing token-level variance and avoiding the high-entropy search space of control-flow loops (loop, block, br\_if, local.set)11.  
> 2. **Execution Safety:** Static constant expressions incur zero risk of runtime traps, infinite loops, or integer overflow exceptions, guaranteeing a non-negative $R\_{\\text{comp}}$ and avoiding severe execution penalties11.  
> 3. **Surrogate Metric Exploitation:** For targets $y\_n$ that exhibit bounded ranges or localized values near $C$, $R\_{\\text{dist}}(P, Y)$ returns a positive baseline return7. The policy gradient identifies that committing to a fixed constant yields a higher expected return per token than exploring complex control structures, where small token mutations frequently trigger total execution failure7.

### **Signal-to-Noise Ratio (SNR) Collapse Mechanism**

The underlying cause of this failure mode can be rigorously analyzed through the gradient decomposition of policy updates11. Consider a batch of inputs $X$ where for each prompt $X\_i$, the model samples $G$ candidate program trajectories $\\{Z\_{i,g}\\}\_{g=1}^G$5. The total policy gradient under baseline-subtracted algorithms (such as GRPO or PPO) decomposes into a task gradient $g\_{\\text{task}}$ and a regularization gradient $g\_{\\text{reg}}$5:

$$\\nabla\_\\theta \\mathcal{J}(\\theta) \= \\underbrace{\\frac{1}{B G} \\sum\_{i=1}^B \\sum\_{g=1}^G \\nabla\_\\theta \\log \\pi\_\\theta(Z\_{i,g} \\mid X\_i) A(X\_i, Z\_{i,g})}\_{g\_{\\text{task}}} \- \\underbrace{\\lambda\_{\\text{reg}} \\nabla\_\\theta D\_{\\text{KL}}(\\pi\_\\theta \\Vert{} \\pi\_0)}\_{g\_{\\text{reg}}}$$  
where $A(X\_i, Z\_{i,g}) \= R(X\_i, Z\_{i,g}) \- \\bar{R}(X\_i)$ represents the advantage, and $\\bar{R}(X\_i) \= \\frac{1}{G} \\sum\_{g=1}^G R(X\_i, Z\_{i,g})$ is the group mean reward5. The magnitude of the task gradient $\\Vert{}g\_{\\text{task}}\\Vert{}$ scales directly with the sample variance of the rewards within the prompt group, $\\mathbb{Var}\_{g}\[R(X\_i, Z\_{i,g})\]$11. When the policy collapses into emitting degenerate constant shortcuts, every sampled rollout $Z\_{i,g}$ produces identical or nearly identical static return values11. As a consequence:

$$\\lim\_{\\mathbb{Var}\_g\[R\] \\to 0} \\Vert{}g\_{\\text{task}}\\Vert{} \= 0$$  
While the task gradient vanishes, the regularization gradient $\\Vert{}g\_{\\text{reg}}\\Vert{}$ (originating from KL divergence or token entropy constraints) remains constant across all inputs11. This creates a catastrophic Signal-to-Noise Ratio (SNR) collapse: updates become entirely dominated by $g\_{\\text{reg}}$, applying an input-agnostic contraction force across all prompt representations11. This contraction erases cross-input functional differences, cementing the policy into an input-independent static template11.

## **2\. Information-Theoretic Regularization and Non-Triviality Objectives**

To prevent the policy gradient from contracting into parameter-ignoring constants, the policy objective must explicitly penalize programs whose outputs are statistically decoupled from their inputs11.

### **Cross-Input Mutual Information Penalties**

A core failure of constant programs is that the mutual information between the input parameter $n$ and the executed output $P(n)$, denoted $I(n; P(n))$, collapses to zero11. To enforce functional input dependence, an explicit mutual information lower-bound constraint is incorporated into the reward signal11:

$$I(n; P(n)) \= H(P(n)) \- H(P(n) \\mid n)$$  
Because exact computation of $I(n; P(n))$ over arbitrary execution spaces is intractable, a batch-estimated sequence-level Mutual Information proxy ($\\text{MI}\_{\\text{proxy}}$) is formulated over a minibatch of $B$ tasks with $N$ input points each11. Let $\\mathbf{e}\_P(n) \= \\text{Embed}(P(n))$ denote a vector representation of the program's output on input $n$11. The cross-input similarity matrix $\\mathbf{S}\_{i,j}$ between task $i$ and task $j$ is computed as11:

$$\\mathbf{S}\_{i,j} \= \\frac{1}{N} \\sum\_{n=0}^{N-1} \\cos\\left(\\mathbf{e}\_{P\_i}(n), \\mathbf{e}\_{P\_j}(n)\\right)$$  
The cross-input mutual information reward is then defined as the negative log-sum-exp of off-diagonal task similarities, forcing outputs across distinct input specifications to be distinguishable11:

$$R\_{\\text{MI}}(P\_i) \= \-\\log \\left( \\frac{1}{B-1} \\sum\_{j \\neq i} \\exp\\left( \\frac{\\mathbf{S}\_{i,j}}{\\tau} \\right) \\right)$$  
where $\\tau \> 0$ is a temperature hyperparameter11. If a policy emits a constant function $P(n) \= C$ regardless of the input task specification, $\\mathbf{S}\_{i,j} \\to 1.0$ for all $j$, driving $R\_{\\text{MI}} \\to \-\\infty$ and heavily penalizing the shortcut trajectory11.

### **Output Variance Penalties and Input Sensitivity Constraints**

Complementary to cross-input mutual information, non-triviality can be strictly enforced within a single task evaluation trajectory by measuring the empirical variance of the output set11:

$$\\mathbb{Var}\_n\[P(n)\] \= \\frac{1}{N} \\sum\_{n=0}^{N-1} \\Big( P(n) \- \\mu\_P \\Big)^2, \\quad \\text{where } \\mu\_P \= \\frac{1}{N} \\sum\_{n=0}^{N-1} P(n)$$  
For tasks where the ground-truth sequence $Y$ is non-constant ($\\mathbb{Var}\_n\[y\_n\] \> 0$), any synthesized program exhibiting $\\mathbb{Var}\_n\[P(n)\] \< \\epsilon$ receives an explicit non-triviality penalty11:

$$R\_{\\text{var}}(P) \= \- \\lambda\_{\\text{var}} \\max\\left(0, \\epsilon\_{\\text{target}} \- \\mathbb{Var}\_n\[P(n)\]\\right)$$  
Furthermore, differential input sensitivity can be assessed via finite-difference approximations over the program's execution trace15. Defining the empirical input sensitivity $\\mathcal{S}\_{\\text{input}}(P)$ as:

$$\\mathcal{S}\_{\\text{input}}(P) \= \\sum\_{n=0}^{N-2} \\left\\vert{} P(n+1) \- P(n) \\right\\vert{}$$  
Programs exhibiting $\\mathcal{S}\_{\\text{input}}(P) \= 0$ when $y\_{n+1} \\neq y\_n$ are gated out of positive reward allocation entirely, breaking the monotonicity of surrogate error distances that allow static constants to accumulate partial reward11.

### **Semantic Diversity Lessons from Evolutionary Program Synthesis**

The problem of semantic stagnation on trivial constants has been extensively studied within evolutionary computation, specifically Semantic Genetic Programming (SGP), PushGP, and Bayesian program learning architectures like DreamCoder3. In traditional Genetic Programming, optimizing aggregate error metrics (such as Mean Squared Error across all test cases) leads to "compromise solutions"—programs that output a static average constant to minimize total penalty across diverse inputs18. To counteract this, modern evolutionary synthesis replaces scalar aggregate fitness with Lexicase Selection and its continuous variant, $\\epsilon$-Lexicase Selection18.

| Selection / Optimization Method | Fitness Evaluation Basis | Mechanism for Preventing Constant Shortcut Collapse | Behavioral Diversity Characteristics |
| :---- | :---- | :---- | :---- |
| **Standard Tournament Selection** \[cite: 18, 19\] | Aggregate Mean Error over all test cases $\\frac{1}{N} \\sum e\_i$18. | None; highly susceptible to static constant attractors that minimize global variance18. | Collapses rapidly; favors mediocre generalists over partial specialists18. |
| **Lexicase Selection** \[cite: 18, 19, 20\] | Unaggregated test cases evaluated in randomized sequence18. | Filters population step-by-step on individual test cases; specialists on specific inputs survive regardless of average error18. | Extremely high; preserves diverse sub-programs that explicitly utilize input parameters18. |
| **$\\epsilon$-Lexicase Selection** \[cite: 20, 21, 23\] | Adaptive error threshold $\\epsilon \= \\text{median}(\\text{error})$ per test case20. | Allows near-elite programs on individual cases to pass filtering steps, accommodating continuous numerical outputs20. | High; prevents premature filtering caused by minor floating-point fluctuations20. |
| **Down-Sampled Lexicase** \[cite: 21, 24, 25\] | Randomly subsampled subset $S \\subset T$ of test cases per generation21. | Dynamic case subsampling alters the fitness landscape every generation, making static shortcuts unsustainable21. | Maximal; forces structural exploration of loop structures to survive shifting test cases21. |
| **DreamCoder Library Learning** \[cite: 16\] | Wake-Sleep Bayesian Compression & Refactoring16. | Refactors reused structural abstractions into a symbolic library during the Sleep phase; static constants are filtered out during semantic deduplication3. | High structural diversity; forces synthesized programs to compose higher-order primitives16. |

Lexicase selection eliminates constant attractors by evaluating candidate programs on individual test cases in a randomized sequence, rather than averaging error vectors18. A candidate program must be strictly elite on the first randomly drawn test case $n\_r$ to survive to the next filtering step18. Because static constants are rarely elite across all individual test points simultaneously, they are eliminated in early filtering steps18. Incorporating down-sampled test-case selection into RLVR minibatches directly mirrors this evolutionary mechanism, ensuring that rewards are conditioned on per-case elite performance rather than batch-averaged surrogate distances21.

## **3\. Potential-Based Reward Shaping and Curriculum Schedules**

A primary reason neuro-symbolic policies collapse to constant shortcuts is that standard dense surrogate rewards ($R\_{\\text{surr}}$) violate Policy Invariance26. When reward shaping is naive, the theoretical optimal policy $\\pi^\* \= \\arg\\max\_\\pi \\mathbb{E}\_{\\pi} \\left\[ \\sum \\gamma^t R\_{\\text{shaped}} \\right\]$ differs from the optimal policy under the true task specification $R\_{\\text{exact}}$, creating artificial global optima corresponding to static constants26.

### **Potential-Based Reward Shaping (PBRS) and Policy Invariance**

As proven by Ng et al. (1999), to guarantee that reward shaping does not alter the optimal policy space of the underlying Markov Decision Process (MDP), the shaping signal $F(s, a, s')$ must be formulated strictly as the difference of a potential function $\\Phi(s)$ evaluated over state transitions26:

$$F(s, a, s') \= \\gamma \\Phi(s') \- \\Phi(s)$$  
where $\\gamma \\in (0, 1\]$ is the MDP discount factor, $s$ is the current generation state (partial code AST), and $s'$ is the next state after emitting token $a$26.  
In program synthesis, defining $\\Phi(s)$ directly over terminal output distance ($R\_{\\text{dist}}$) violates potential-based invariants because code construction is a directional, autoregressive sequence where partial AST states do not naturally map to valid execution outputs without heuristic padding7. To enforce valid potential-based shaping, $\\Phi(s)$ must be formulated over the Abstract Syntax Tree (AST) completion state26:

$$\\Phi(s) \= \\begin{cases} 0 & \\text{if } s \= s\_0 \\text{ (root context)} \\\\ \\phi\_{\\text{comp}}(s) \+ \\phi\_{\\text{bind}}(s) & \\text{if } s \\text{ is an intermediate AST state} \\\\ \\Phi\_{\\text{terminal}}(P) & \\text{if } s \\text{ is a completed program } P \\end{cases}$$  
where $\\phi\_{\\text{comp}}(s) \> 0$ indicates valid syntactic compilation of the partial token stream, $\\phi\_{\\text{bind}}(s) \> 0$ is a static potential allocated when the AST explicitly binds the input variable parameter $n$ within a dynamic control loop, and $\\Phi\_{\\text{terminal}}(P) \= R\_{\\text{exact}}(P, Y)$6.  
Because $\\sum\_{t=0}^{T-1} F(s\_t, a\_t, s\_{t+1}) \= \\gamma^T \\Phi(s\_T) \- \\Phi(s\_0)$, the cumulative shaped reward over an entire program trajectory telescopes cleanly:

$$R\_{\\text{total}}(P) \= R\_{\\text{exact}}(P, Y) \+ \\gamma^T \\Phi\_{\\text{terminal}}(P) \- \\Phi(s\_0)$$  
This telescoping sum guarantees that the relative ordering of arbitrary candidate programs under $R\_{\\text{total}}$ is identical to their ordering under $R\_{\\text{exact}}$, mathematically eliminating artificial local attractors caused by heuristic surrogate metrics26.

### **Dynamic Annealing and Gating Schedules**

To transition the policy from soft exploration to strict exact execution without inducing collapse, the reward framework employs an adaptive curriculum schedule28. Dense surrogate metrics ($R\_{\\text{dist}}, R\_{\\text{prefix}}$) are dynamically annealed based on the policy's rolling success rate $\\mathcal{S}\_{\\text{roll}}$ over a sliding window of $W$ training iterations28:

$$w\_{\\text{surr}}(t) \= w\_0 \\cdot \\left(1 \- \\tanh\\left( \\kappa \\cdot \\mathcal{S}\_{\\text{roll}}(t) \\right)\\right)$$

$$w\_{\\text{exact}}(t) \= 1.0 \- w\_{\\text{surr}}(t)$$  
As policy competence increases ($\\mathcal{S}\_{\\text{roll}} \\to 1.0$), surrogate weights decay smoothly to zero, leaving strictly verifiable exact matching rewards6. Furthermore, a Hard Gating Threshold is established: $R\_{\\text{dist}}(P, Y)$ is set to zero unless the program $P$ satisfies a baseline structural complexity check—specifically containing at least one conditional branch or iterative loop operator (loop, block, br\_if). By enforcing this structural gating, unparameterized static constants receive $R\_{\\text{dist}} \= 0$ regardless of their continuous numerical proximity to $Y$, preventing them from serving as viable stepping stones in early training7.

## **4\. Minimum Description Length and Algorithmic Information Penalties**

To restrict the policy from generating overly simplistic unparameterized bytecodes, concepts from Algorithmic Information Theory (AIT) and Minimum Description Length (MDL) principles are incorporated directly into the policy optimization objective29.

### **Kolmogorov Complexity Approximations in Code Synthesis**

The Kolmogorov complexity $K(P)$ of a program $P$ is defined as the length of the shortest binary string that outputs $P$ on a universal Turing machine30. Because $K(P)$ is non-computable, practical approximations rely on lossless compression measurements over the serialized Abstract Syntax Tree $\\text{ser}(\\text{AST}(P))$34:

$$\\tilde{K}(P) \= \\left\\vert{} \\text{Compress}\\Big(\\text{ser}(\\text{AST}(P))\\Big) \\right\\vert{}$$  
where $\\text{Compress}(\\cdot)$ denotes a standard Lempel-Ziv (LZ4 or zlib) compression operator34. For a program synthesis specification $(X, Y)$, the ideal MDL objective minimizes the joint description length of the program model and the execution residuals29:

$$\\tilde{K}(P) \= \\left\\vert{} \\text{Compress}\\Big(\\text{ser}(\\text{AST}(P))\\Big) \\right\\vert{}$$  
where $K(Y \\mid P(X)) \= \\sum\_{n=0}^{N-1} \\lceil \\log\_2(1 \+ \\vert{}P(n) \- y\_n\\vert{}) \\rceil$29. A degenerate constant program $P\_{\\text{const}}$ exhibits an extremely low model description length $\\tilde{K}(P\_{\\text{const}}) \\approx O(1)$29. However, when evaluated on non-trivial sequences where $Y$ is dynamically generated by a parameter $n$, the data residual length $K(Y \\mid P\_{\\text{const}}(X))$ scales linearly with sequence length $N \\cdot \\log\_2(\\text{Range}(Y))$, causing the total MDL cost to explode relative to a compact parameterized loop program $P\_{\\text{loop}}$:

$$\\mathcal{L}\_{\\text{MDL}}(P; X, Y) \= \\underbrace{\\tilde{K}(P)}\_{\\text{Model Description Length}} \+ \\lambda\_{\\text{res}} \\underbrace{K(Y \\mid P(X))}\_{\\text{Data Residual Length}}$$

### **AST Entropy and Syntactic Regularization**

In addition to compression-based description lengths, syntactic entropy is regularized at the AST representation level11. Let $\\mathcal{T}(P)$ denote the set of non-terminal and terminal nodes in the parsed AST of program $P$11. The AST token distribution entropy $H(\\mathcal{T}(P))$ is formulated over the dynamic node types (such as constants, variables, arithmetic operators, and control flow blocks)11:

$$H(\\mathcal{T}(P)) \= \-\\sum\_{v \\in \\text{NodeTypes}} p(v \\mid P) \\log\_2 p(v \\mid P)$$  
where $p(v \\mid P) \= \\frac{\\text{Count}(v, \\mathcal{T}(P))}{\\vert{}\\mathcal{T}(P)\\vert{}}$11. Degenerate static programs consisting of a single return constant exhibit $H(\\mathcal{T}(P\_{\\text{const}})) \\approx 0$11. To penalize low-complexity syntactic shortcuts, a non-linear AST entropy penalty $\\mathcal{R}\_{\\text{AST}}(P)$ is added to the loss function11:

$$H(\\mathcal{T}(P)) \= \-\\sum\_{v \\in \\text{NodeTypes}} p(v \\mid P) \\log\_2 p(v \\mid P)$$  
where $\\mathbb{I}\\left( \\text{VarBind}(P) \= \\emptyset \\right)$ is an indicator function evaluating to $1$ if the program AST contains zero occurrences of the input parameter token $n$11.

### **Consolidated Regularized Policy Gradient Loss**

Integrating the mutual information proxy, potential-based shaping, and MDL syntactic penalties yields the overall regularized policy objective11:

$$\\mathcal{L}\_{\\text{total}}(\\theta) \= \\mathcal{L}\_{\\text{RLVR}}(\\theta) \- \\lambda\_{\\text{MI}} R\_{\\text{MI}}(P) \+ \\lambda\_{\\text{MDL}} \\mathcal{L}\_{\\text{MDL}}(P) \+ \\lambda\_{\\text{AST}} \\mathcal{R}\_{\\text{AST}}(P)$$

$$\\mathcal{L}\_{\\text{RLVR}}(\\theta) \= \-\\mathbb{E}\_{\\mathcal{S} \\sim \\mathcal{D}, P \\sim \\pi\_\\theta} \\left\[ \\min\\left( r\_\\theta(P) A\_{\\text{PBRS}}, \\text{clip}(r\_\\theta(P), 1-\\epsilon, 1+\\epsilon) A\_{\\text{PBRS}} \\right) \\right\]$$  
where $r\_\\theta(P) \= \\frac{\\pi\_\\theta(P \\mid \\mathcal{S})}{\\pi\_{\\theta\_{\\text{old}}}(P \\mid \\mathcal{S})}$, and $A\_{\\text{PBRS}}$ is the advantage computed under the Potential-Based Reward Shaping signal5.

## **5\. Diversity-Promoting Exploration Architectures**

When standard entropy penalties fail to force autoregressive models out of local constant attractors, structural exploration paradigms must be integrated into the sampling pipeline11.

### **Quality-Diversity (MAP-Elites) over Execution Semantics**

Standard policy gradient methods optimize solely for expected return, causing all population trajectories to converge onto the single easiest high-reward path—the static constant35. Quality-Diversity (QD) algorithms, such as MAP-Elites, overcome this by maintaining a multi-dimensional archive of solutions partitioned across explicit Behavior Descriptors (BDs)36.  
In neuro-symbolic synthesis, the Behavior Descriptor space $\\mathcal{B}(P)$ is defined over execution semantics rather than token syntax36:

$$\\mathcal{B}(P) \= \\Big\[ \\text{BD}\_1(P), \\; \\text{BD}\_2(P), \\; \\text{BD}\_3(P) \\Big\]$$

> * **$\\text{BD}\_1(P) \= H\\big(\\{P(n)\\}\_{n=0}^{N-1}\\big)$:** Execution output entropy across evaluation points11.  
> * **$\\text{BD}\_2(P) \= \\text{InstrCount}(P)$:** Total executed instruction count (distinguishing zero-cost constants from multi-cycle loops).  
> * **$\\text{BD}\_3(P) \= \\text{BranchCoverage}(P)$:** Fraction of dynamic basic blocks traversed during execution.

The feature space $\\mathcal{B}$ is discretized into a tessellated $D$-dimensional grid (e.g., $10 \\times 10 \\times 10$ cells)36. The archive maintains only the highest-performing program (Elite) per cell36. During RL sampling, prompts are paired with parent programs mutated from under-populated cells in the archive36. Because low-entropy static constants inhabit a single isolated cell ($\\text{BD}\_1 \\approx 0, \\text{BD}\_2 \\approx 1, \\text{BD}\_3 \= 0$), the MAP-Elites archive saturates that cell immediately and diverts all subsequent selection pressure toward filling empty cells corresponding to higher execution complexity and output entropy36.

### **Generative Flow Networks (GFlowNets) for Diverse Code Generation**

While traditional RL maximizes expected cumulative reward—a process prone to mode collapse—Generative Flow Networks (GFlowNets) train a stochastic policy to sample discrete objects (programs $P$) with probability proportional to a non-negative reward signal12:

$$P\_{\\theta}(P) \\propto R(P)$$  
By treating code generation as a flow network over a Directed Acyclic Graph (DAG) of partial AST states, GFlowNets naturally discover and preserve multiple distinct structural modes, preventing the policy from collapsing into a single constant shortcut12.  
Rather than operating on unsemantic BPE tokens, the forward policy $P\_F(s\_{t+1} \\mid s\_t; \\theta)$ and backward policy $P\_B(s\_t \\mid s\_{t+1}; \\theta)$ operate directly at the AST node level or complete block level35. A trajectory $\\tau \= (s\_0 \\to s\_1 \\to \\dots \\to s\_T \= P)$ represents the sequential derivation of an AST35. The policy is trained using the Trajectory Balance (TB) objective35:

$$\\mathcal{L}\_{\\text{TB}}(\\tau; \\theta) \= \\left( \\log Z\_\\theta \+ \\sum\_{t=0}^{T-1} \\log P\_F(s\_{t+1} \\mid s\_t; \\theta) \- \\log R(P) \- \\sum\_{t=0}^{T-1} \\log P\_B(s\_t \\mid s\_{t+1}; \\theta) \\right)^2$$  
where $Z\_\\theta$ is a learned scalar estimating the total state flow (partition function)40. To optimize credit assignment along long AST generation paths, the Subtrajectory Balance (SubTB) loss is applied across sub-paths $s\_i \\to \\dots \\to s\_j$35:

$$\\mathcal{L}\_{\\text{TB}}(\\tau; \\theta) \= \\left( \\log Z\_\\theta \+ \\sum\_{t=0}^{T-1} \\log P\_F(s\_{t+1} \\mid s\_t; \\theta) \- \\log R(P) \- \\sum\_{t=0}^{T-1} \\log P\_B(s\_t \\mid s\_{t+1}; \\theta) \\right)^2$$  
Because GFlowNets enforce flow consistency across the entire generation graph, high rewards assigned to parameterized loop solutions generate backward flow that propagates credit to early structural choices (e.g., emitting loop or param $n tokens), maintaining high probability mass on complex paths even if constant programs also achieve low partial reward35.

## **6\. Implementation Blueprint and Unified Optimization Pipeline**

The following comparative synthesis evaluates the principal technical strategies for eliminating degenerate shortcut collapse, detailing their mathematical guarantees, implementation requirements, and operational trade-offs13.

| Strategy | Underlying Mechanism | Primary Guarantee | Implementation Complexity | Computational Overhead | Primary Failure Risks & Mitigations |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Cross-Input Mutual Information ($R\_{\\text{MI}}$)** \[cite: 11, 13\] | Penalizes off-diagonal cross-task sequence similarities in batch execution embeddings11. | Guarantees $I(n; P(n)) \> 0$; forces output distinguishability11. | Medium; requires tensor parallel execution logging11. | Low (+5% batch compute overhead)11. | High temperature $\\tau$ causes gradient instability; stabilize via EMA z-score scaling11. |
| **Potential-Based Reward Shaping (PBRS)** \[cite: 26, 27\] | Formulates dense shaping strictly as $\\gamma \\Phi(s') \- \\Phi(s)$ over AST states26. | **Policy Invariance**: Optimal policy set under shaped reward matches ground truth26. | High; requires formal AST completion state mapping26. | Negligible (\<1% compute overhead)26. | Poor potential design ($\\Phi$) leads to zero gradient; pair with variable binding potentials26. |
| **Minimum Description Length (MDL) / AST Loss** \[cite: 29, 31\] | Combines lossless AST compression penalties with AST token entropy constraints29. | Algorithmic Information Theory bound on data description length29. | Medium; requires fast C-extension for zlib AST serialization34. | Low (+2% training step latency). | Over-penalizing model length suppresses valid loop unrolling; tune $\\lambda\_{\\text{MDL}}$ carefully29. |
| **Dynamic Down-Sampled Lexicase RLVR** \[cite: 21, 24, 25\] | Replaces aggregate loss with per-case elite filtering on dynamic test sub-batches18. | Prevents compromise constant attractors; selects per-input specialists18. | Low; modify reward reduction in loss kernel18. | None (0% added overhead)24. | Small sample size increases reward variance; stabilize with baseline subtraction11. |
| **Quality-Diversity (MAP-Elites Archive)** \[cite: 36, 37\] | Maintains multi-cell program archive across output entropy and instruction counts36. | Uniform coverage across execution behavior descriptor space $\\mathcal{B}$36. | High; requires custom dynamic sampling buffer and execution tracer36. | Moderate (+15% rollout evaluation time)36. | Bin discretization granularity affects archive expansion; use adaptive Voronoi tessellation36. |
| **GFlowNet Fine-Tuning (TB / SubTB)** \[cite: 35, 40\] | Trains DAG flow policy to sample program ASTs with probability $P(P) \\propto R(P)$35. | Amortized sampling across all high-reward modes; eliminates mode collapse12. | Very High; requires forward/backward policy head and flow estimator $Z\_\\theta$35. | High (+30% backprop computation latency)35. | Trajectory balance variance on long sequences; apply SubTB over local AST sub-trees35. |

### **Unified Robust Program Synthesis Pipeline**

To integrate these defenses into a unified training workflow, execution follows a four-stage optimization loop during each training iteration11:

> 1. **SNR-Aware Sampling and Filtering Stage:** For each task specification $S\_i$ in a batch, the policy samples $G$ candidate AST trajectories5. The executor evaluates the programs on input vectors, measuring the within-group reward variance $\\mathbb{Var}\_g\[R\_{\\text{exact}}(P\_{i,g})\]$11. Tasks exhibiting variance below $\\sigma^2\_{\\text{min}}$ are filtered out prior to gradient computation, ensuring updates are not driven by zero-variance regularization noise11.  
> 2. **Multi-Objective Reward Assignment Stage:** Surviving trajectories are assigned composite rewards incorporating telescoping Potential-Based Reward Shaping ($R\_{\\text{PBRS}}$), batch cross-input mutual information ($R\_{\\text{MI}}$), and lossless AST compression length penalties ($\\mathcal{L}\_{\\text{MDL}}$)11.  
> 3. **Quality-Diversity Archive Update:** The executing programs are evaluated against their behavioral descriptors (output entropy, instruction count, and branch coverage)36. Programs that set new performance elites within their respective feature grid cells are written to the MAP-Elites archive buffer36.  
> 4. **Flow-Balanced Parameter Update:** Policy weights are updated by minimizing the GFlowNet Subtrajectory Balance loss ($\\mathcal{L}\_{\\text{SubTB}}$) alongside AST non-triviality entropy penalties ($\\mathcal{R}\_{\\text{AST}}$), updating parameters via AdamW while synchronizing reference models via exponential moving averages11.

## **7\. Conclusions and Strategic Recommendations**

Degenerate constant shortcut collapse in neuro-symbolic program synthesis is an inevitable optimization outcome when autoregressive policies are trained with naive, dense surrogate rewards under dynamic grammar masking11. Because static constants eliminate execution traps and yield non-zero surrogate returns without incurring entropy penalties, they create a catastrophic drop in within-group reward variance11. This drops the task gradient magnitude $\\|g\_{\\text{task}}\\|$ to near zero, causing the static regularization gradient $g\_{\\text{reg}}$ to dominate updates and contract the policy into an input-agnostic prior11.  
To permanently prevent constant shortcut collapse and achieve robust out-of-distribution programmatic generalization, implementation pipelines should adopt the following primary design practices:

> 1. **Replace Heuristic Surrogate Metrics with Potential-Based Reward Shaping (PBRS):** Unshaped output distance metrics ($R\_{\\text{dist}}$) destroy policy invariance26. Dense rewards must be derived strictly through telescoping potential functions $\\gamma \\Phi(s') \- \\Phi(s)$ defined over structural AST completion and explicit variable binding states26.  
> 2. **Mandate Cross-Input Mutual Information Regularization:** Integrate batch-level sequence mutual information proxies ($R\_{\\text{MI}}$) into the reward loop11. Penalizing cross-task output vector similarities forces the model to maintain strong functional input dependence $I(n; P(n)) \> 0$ across all training steps11.  
> 3. **Incorporate Structural Gating and Lexicase Filtering:** Enforce hard gating thresholds that zero out surrogate reward allocation for any code trajectory lacking parameterized variable bindings (param $n) or control-flow primitives (loop, br\_if)7. Replace scalar aggregate loss reductions with down-sampled lexicase selection to preserve specialized sub-programs18.  
> 4. **Transition Optimization to GFlowNet Distribution Matching:** Shift away from standard RL reward-maximization algorithms (PPO/GRPO) which inherently collapse onto single deterministic modes12. Train autoregressive models as Generative Flow Networks using Subtrajectory Balance objectives, enabling the network to learn stochastic policies that sample diverse algorithmic structures proportional to their true verifiable rewards35.

#### **Works cited**

> 1. Chapter 3: Reinforcement Learning of Large Language Models, [https://ernestryu.com/courses/RL-LLM/chapter3.pdf](https://ernestryu.com/courses/RL-LLM/chapter3.pdf)  
> 2. Akhilesh Deepak Gotmare \- alphaXiv, [https://www.alphaxiv.org/@akhilesh-deepak-gotmare](https://www.alphaxiv.org/@akhilesh-deepak-gotmare)  
> 3. Program Synthesis via Test-Time Transduction \- arXiv, [https://arxiv.org/html/2509.17393v2](https://arxiv.org/html/2509.17393v2)  
> 4. nathanael-fijalkow/DeepSynth \- GitHub, [https://github.com/nathanael-fijalkow/DeepSynth](https://github.com/nathanael-fijalkow/DeepSynth)  
> 5. LaSeR: Reinforcement Learning with Last-Token Self-Rewarding, [https://openreview.net/forum?id=1OhgEmix20](https://openreview.net/forum?id=1OhgEmix20)  
> 6. Awesome RLVR — Reinforcement Learning with Verifiable Rewards, [https://github.com/opendilab/awesome-RLVR](https://github.com/opendilab/awesome-RLVR)  
> 7. A Review of Reward Program Synthesis, Multimodal Feedback, a, [https://www.preprints.org/frontend/manuscript/fb6b219f167f1a09c9f21d8c9fb0e3ea/download\_pub](https://www.preprints.org/frontend/manuscript/fb6b219f167f1a09c9f21d8c9fb0e3ea/download_pub)  
> 8. Algorithms and Applications of Explainable Machine Learning, [https://search.proquest.com/openview/877da46c5acf0246a8697c55566e61ad/1?pq-origsite=gscholar\&cbl=18750\&diss=y](https://search.proquest.com/openview/877da46c5acf0246a8697c55566e61ad/1?pq-origsite=gscholar&cbl=18750&diss=y)  
> 9. Interpreting Model-Agnostic Counterfactual Explanations of a Deep, [https://iris.uniroma1.it/retrieve/db013d9c-1e30-442d-b5ea-58674391987d/Chen\_postprint\_Explain\_2024.pdf.pdf](https://iris.uniroma1.it/retrieve/db013d9c-1e30-442d-b5ea-58674391987d/Chen_postprint_Explain_2024.pdf.pdf)  
> 10. Interpreting Model-Agnostic Counterfactual Explanations of a Deep, [https://www.researchgate.net/publication/365739708\_Explain\_the\_Explainer\_Interpreting\_Model-Agnostic\_Counterfactual\_Explanations\_of\_a\_Deep\_Reinforcement\_Learning\_Agent](https://www.researchgate.net/publication/365739708_Explain_the_Explainer_Interpreting_Model-Agnostic_Counterfactual_Explanations_of_a_Deep_Reinforcement_Learning_Agent)  
> 11. RAGEN-2: Reasoning Collapse in Agentic RL \- arXiv, [https://arxiv.org/html/2604.06268v1](https://arxiv.org/html/2604.06268v1)  
> 12. Structurally Valid Log Generation using FSM-GFlowNets \- arXiv, [https://arxiv.org/pdf/2510.26197](https://arxiv.org/pdf/2510.26197)  
> 13. Integrating Reinforcement Learning with Visual Generative Models, [https://arxiv.org/html/2508.10316v2](https://arxiv.org/html/2508.10316v2)  
> 14. Statistical limits and conditional complexity in real-world ... \- Frontiers, [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1847643/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1847643/full)  
> 15. Track: Poster Session 3 \- ICML 2026, [https://icml.cc/virtual/2026/session/68687](https://icml.cc/virtual/2026/session/68687)  
> 16. Lilo: Learning Interpretable Libraries by Compressing and ... \- arXiv, [https://arxiv.org/html/2310.19791v4](https://arxiv.org/html/2310.19791v4)  
> 17. Automated Program Synthesis \- Emergent Mind, [https://www.emergentmind.com/topics/automated-program-synthesis](https://www.emergentmind.com/topics/automated-program-synthesis)  
> 18. The Impact of Hyperselection on Lexicase Selection \- CMAP, [http://www.cmap.polytechnique.fr/\~nikolaus.hansen/proceedings/2016/GECCO/proceedings/p717.pdf](http://www.cmap.polytechnique.fr/~nikolaus.hansen/proceedings/2016/GECCO/proceedings/p717.pdf)  
> 19. (PDF) Solving Uncompromising Problems With Lexicase Selection, [https://www.researchgate.net/publication/276254601\_Solving\_Uncompromising\_Problems\_With\_Lexicase\_Selection](https://www.researchgate.net/publication/276254601_Solving_Uncompromising_Problems_With_Lexicase_Selection)  
> 20. A probabilistic and multi-objective analysis of lexicase selection and, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9453780/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9453780/)  
> 21. Down-Sampled Epsilon-Lexicase Selection for Real-World Symbolic, [https://arxiv.org/pdf/2302.04301](https://arxiv.org/pdf/2302.04301)  
> 22. A Comprehensive Survey on Lexicase Selection, [https://download.uni-mainz.de/RePEc/pdf/Discussion\_Paper\_2605.pdf](https://download.uni-mainz.de/RePEc/pdf/Discussion_Paper_2605.pdf)  
> 23. A Performance Analysis of Lexicase-Based and Traditional ... \- arXiv, [https://arxiv.org/html/2407.21632v2](https://arxiv.org/html/2407.21632v2)  
> 24. Problem-Solving Benefits of Down-Sampled Lexicase Selection, [https://direct.mit.edu/artl/article/27/3%E2%80%934/183/106924/Problem-Solving-Benefits-of-Down-Sampled-Lexicase](https://direct.mit.edu/artl/article/27/3%E2%80%934/183/106924/Problem-Solving-Benefits-of-Down-Sampled-Lexicase)  
> 25. The Problem Solving Benefits of Down-sampling Vary by Selection, [https://par.nsf.gov/servlets/purl/10463873](https://par.nsf.gov/servlets/purl/10463873)  
> 26. (PDF) Improving the Effectiveness of Potential-Based Reward, [https://www.researchgate.net/publication/388657925\_Improving\_the\_Effectiveness\_of\_Potential-Based\_Reward\_Shaping\_in\_Reinforcement\_Learning](https://www.researchgate.net/publication/388657925_Improving_the_Effectiveness_of_Potential-Based_Reward_Shaping_in_Reinforcement_Learning)  
> 27. Foundation-Model-Assisted Reward Design for Reinforcement, [https://www.preprints.org/manuscript/202608.0779](https://www.preprints.org/manuscript/202608.0779)  
> 28. Policy Improvement Reinforcement Learning \- arXiv, [https://arxiv.org/html/2604.00860v5](https://arxiv.org/html/2604.00860v5)  
> 29. Single-pass Adaptive Image Tokenization for Minimum Program, [https://neurips.cc/virtual/2025/poster/118875](https://neurips.cc/virtual/2025/poster/118875)  
> 30. Model Selection Based on Minimum Description Length, [https://iri.columbia.edu/\~tippett/cv\_papers/Grunwald.pdf](https://iri.columbia.edu/~tippett/cv_papers/Grunwald.pdf)  
> 31. Single-pass Adaptive Image Tokenization for Minimum Program, [https://arxiv.org/html/2507.07995v1](https://arxiv.org/html/2507.07995v1)  
> 32. (PDF) Learning Theory from the Viewpoint of Algorithmic Information, [https://www.researchgate.net/publication/394846987\_Learning\_Theory\_from\_the\_Viewpoint\_of\_Algorithmic\_Information\_Theory\_Kolmogorov\_Complexity\_Meets\_Kernel\_Methods](https://www.researchgate.net/publication/394846987_Learning_Theory_from_the_Viewpoint_of_Algorithmic_Information_Theory_Kolmogorov_Complexity_Meets_Kernel_Methods)  
> 33. Quantum Kolmogorov Complexity Based on Classical Descriptions, [https://ir.cwi.nl/pub/2073/2073D.pdf](https://ir.cwi.nl/pub/2073/2073D.pdf)  
> 34. The Algorithmic Regulator \- MDPI, [https://www.mdpi.com/1099-4300/28/3/257](https://www.mdpi.com/1099-4300/28/3/257)  
> 35. diverse llm mathemati \- arXiv, [https://arxiv.org/pdf/2504.19981](https://arxiv.org/pdf/2504.19981)  
> 36. DEI: Diversity in Evolutionary Inference for Quality-Diversity Search, [https://arxiv.org/html/2605.27130v1](https://arxiv.org/html/2605.27130v1)  
> 37. QDTraj: Exploration of Diverse Trajectory Primitives for ... \- Bytez, [https://bytez.com/docs/arxiv/2604.22551/paper](https://bytez.com/docs/arxiv/2604.22551/paper)  
> 38. List of papers | Quality-Diversity optimisation algorithms, [https://quality-diversity.github.io/papers.html](https://quality-diversity.github.io/papers.html)  
> 39. Unsupervised and Problem-Agnostic Quality-Diversity Optimization, [https://arxiv.org/html/2504.08057v3](https://arxiv.org/html/2504.08057v3)  
> 40. Trajectory Balance: Improved Credit Assignment in GFlowNets, [https://www.researchgate.net/publication/401451811\_Trajectory\_Balance\_Improved\_Credit\_Assignment\_in\_GFlowNets](https://www.researchgate.net/publication/401451811_Trajectory_Balance_Improved_Credit_Assignment_in_GFlowNets)  
> 41. Training LLMs for Divergent Problem Solving with Minimal Examples, [https://arxiv.org/html/2406.05673v5](https://arxiv.org/html/2406.05673v5)  
> 42. Scalable and Cost-Efficient de Novo Template-Based Molecular, [https://arxiv.org/html/2506.19865v1](https://arxiv.org/html/2506.19865v1)