# **Bootstrapping Neural Code Generation Models Under Sparse Execution Rewards: A Neuro-Symbolic Synthesis Review**

## **Executive Summary of Fundamental Paradigms**

Synthesizing executable algorithmic routines from high-level specifications—such as generating low-level WebAssembly Text (WAT) stack bytecode from integer sequence observations—presents a high-entropy, combinatorial search space1. When training deep autoregressive models under pure reinforcement learning (RL), such as Proximal Policy Optimization (PPO) or Group Relative Policy Optimization (GRPO), the agent relies on binary execution rewards where $R \\in \\{+1, \-1\\}$ (or $R \\in \\{1, 0\\}$)4. In this formulation, a reward of $+1$ is granted strictly if the synthesized bytecode executes without runtime error and generates exact matching terms across all evaluated domain indices $n \\in \\{0, 1, \\dots, N-1\\}$5. If any generated term mismatches or if the bytecode triggers an execution trap (e.g., stack underflow, invalid memory access, or non-termination), the trajectory receives a reward of $-1$5.  
Under a random policy initialization $\\pi\_{\\theta\_0}$, the probability of sampling a syntactically valid and semantically correct sequence of stack instructions that computes a complex $N$-term recurrence relation from scratch is effectively zero ($P(R \= \+1) \\approx 0$)5. This sparse-reward setting induces a fundamental optimization failure known as zero-advantage collapse6. In group-based policy optimization algorithms like GRPO, advantages are estimated relative to the mean and standard deviation of rewards within a batch of $G$ sampled completions $\\{y\_1, y\_2, \\dots, y\_G\\}$ for a given prompt $q$6:

$$\\hat{A}\_i \= \\frac{r\_i \- \\mu\_{\\mathbf{r}}}{\\sigma\_{\\mathbf{r}} \+ \\epsilon}$$  
When every rollout in a sample group fails ($r\_i \= \-1$ for all $i \\in \\{1, \\dots, G\\}$), the empirical mean equals the individual rewards ($\\mu\_{\\mathbf{r}} \= \-1$) and the intra-group reward variance collapses to zero ($\\sigma\_{\\mathbf{r}} \= 0$)6. Consequently, the computed advantage $\\hat{A}\_i$ for every completion evaluates to zero, yielding a vanishing policy gradient $\\nabla\_\\theta J(\\theta) \= 0$7. The model receives no directional learning signal, the computational budget consumed during rollout sampling is wasted, and the policy plateaus permanently in a state of exploration starvation7.  
To bypass this cold-start bottleneck, the academic literature in neuro-symbolic program synthesis, deep reinforcement learning, and formal verification has established four primary foundational paradigms:

> 1. **Supervised Demonstration & Forward Synthetic Warmup**: Grounding the model's policy prior to RL by pretraining or fine-tuning on forward-generated synthetic programs paired with their execution outputs10.  
> 2. **Expert Iteration (ExIt) & Ground-Truth Trajectory Injection**: Coupling on-policy generation with external symbolic verifiers, search buffers, or conditional reference injections (e.g., S-GRPO, MAPO) to force non-zero advantage signals5.  
> 3. **Minimum Description Length (MDL) & Search-Guided Proposal Generators**: Integrating symbolic program search (e.g., enumerative search, stack-based genetic programming, or wake-sleep library learning as in DreamCoder) to discover structural seeds and abstractions13.  
> 4. **Execution Semantics & Step-Level Advantage Alignment**: Transforming the monolithic trajectory-level binary signal into intermediate dense execution traces, variable state alignments, or dynamic hint-guided reward distributions9.

## **Analysis of Bootstrapping and Exploration Warmup Strategies**

Overcoming the zero-percent initial success rate requires hybridizing parametric policy learning with non-parametric symbolic exploration6. Three distinct strategy families dominate the program synthesis literature, each exhibiting clear trade-offs regarding sample complexity, structural generalization, and operational overhead5.

### **Supervised Fine-Tuning Warmup on Synthetic Demonstrations**

Supervised Fine-Tuning (SFT) establishes an initial policy distribution $\\pi\_{\\theta\_{\\text{SFT}}}$ capable of producing syntactically sound programs4. In domains where real-world problem-solution pairs are scarce—such as synthesizing WebAssembly stack bytecode for integer sequences—synthetic demonstration generation provides a scalable alternative10.  
This approach constructs a forward dataset by sampling random, well-typed programs $P \\sim \\text{Grammar}(\\text{WAT})$ from the domain-specific language (DSL) rules10. These programs are executed in a sandboxed interpreter across inputs $n \\in \\{0, 1, \\dots, N-1\\}$ to obtain sequence vectors $Y \= \[f\_P(0), f\_P(1), \\dots, f\_P(N-1)\]$11. The Transformer encoder-decoder is then trained via Maximum Likelihood Estimation (MLE) to minimize the cross-entropy loss over the target tokens17:

$$\\mathcal{L}\_{\\text{SFT}}(\\theta) \= \-\\sum\_{t=1}^{\\vert{}P\\vert{}} \\log \\pi\_\\theta(p\_t \\mid p\_{\<t}, Y)$$  
While SFT rapidly instills syntactic fluency and basic arithmetic stack manipulations (e.g., local.get, i32.add, i32.mul), it suffers from severe distributional bias10. Uniformly sampled random programs over-represent simple linear expressions or rapidly diverging exponential functions, failing to cover complex algorithmic patterns like nested loops, conditional branching, or stateful memory shifts10. Furthermore, MLE forces the model to imitate specific syntactic representations rather than optimizing for functional equivalence, penalizing alternative bytecode sequences that yield identical execution outputs18.

### **Expert Iteration, Self-Taught Reasoner, and Trajectory Injection**

Expert Iteration (ExIt) and its derivatives (STaR, ReST, S-GRPO) construct an adaptive bootstrap loop by leveraging an external execution verifier as an expert oracle6. During each iteration, the current policy samples $K$ completions for a set of problem specifications19. Completions that pass verification ($R \= \+1$) are appended to an elite demonstration memory buffer $\\mathcal{D}\_{\\text{mem}}$, which is subsequently used for policy retraining via supervised behavior cloning or policy gradient updates5.  
To prevent zero-advantage collapse when the initial pass rate across all $K$ samples is strictly zero percent, advanced algorithms modify the sampling process:

> * **Supervised Group Relative Policy Optimization (S-GRPO)**: Integrates Conditional Ground-Truth Trajectory Injection (CGI)6. When a binary verifier detects that an entire rollout group of size $G$ has failed ($r\_i \= \-1 \\ \\forall i$), S-GRPO artificially injects a known reference trajectory $y^\*\_{\\text{gt}}$ into the candidate pool and assigns it a maximal reward $r\_{\\text{gt}} \= \+1$6. This guarantees $\\sigma\_{\\mathbf{r}} \> 0$, yielding a positive advantage $\\hat{A}\_{\\text{gt}} \\gg 0$ for the reference solution and negative advantages $\\hat{A}\_{\\text{gen}} \< 0$ for the failed rollouts, converting an uninformative failure batch into an informative imitation step6.  
> * **Memory-Augmented Policy Optimization (MAPO)**: Maintains an active memory buffer of historical success trajectories for each problem context5. MAPO decomposes the policy gradient objective into separate expectations over trajectories inside and outside the memory buffer, utilizing memory weight clipping to bound gradient variance and prevent the policy from forgetting rare success paths5.

### **Symbolic Search and Minimum Description Length Proposal Generators**

Rather than relying purely on neural sampling, neuro-symbolic frameworks deploy explicit symbolic search algorithms to generate seed solutions1. Systems like DreamCoder operationalize the Minimum Description Length (MDL) principle within a wake-sleep Bayesian learning architecture13.  
During the waking search phase, a symbolic enumerative engine (or stack-based genetic program search engine) explores the program space bounded by type constraints, prioritizing programs with lower description entropy13:

$$\\text{Cost}(P \\mid Y) \= \-\\log P(P) \- \\sum\_{n=0}^{N-1} \\log P(y\_n \\mid P, n)$$  
When a search solver discovers a valid program $P^\*$ satisfying $P^\*(n) \= y\_n$, the sleep phase engages two neural mechanisms:

> 1. **Abstraction (Library Learning)**: The system refactors common sub-expressions found across discovered solutions into new primitive operations within the DSL, effectively compressing the search depth for future tasks14.  
> 2. **Generative Neural Recognition Training**: The neural decoder policy is trained on the discovered solutions to direct future enumerative search toward high-probability programmatic regions13.

| Warmup Strategy | Core Mechanism | Sample Complexity | Exploration Depth | Algorithmic Overhead | Vulnerability to Mode Collapse | Initial Pass@K Capability |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Synthetic SFT (Forward Execution)** | MLE on generated $(P \\to Y)$ pairs10. | Low (pure offline batch pretraining)17. | Low (bounded by random program generator distributions)10. | Minimal (standard NTP cross-entropy loss)4. | High (overfits to simple syntactic idioms)10. | Moderate for short programs; low for complex loops10. |
| **Expert Iteration (ExIt / ReST / STaR)** | On-policy sampling filtered by execution verifier19. | Moderate to High (requires continuous online sampling)6. | Moderate (constrained by current policy entropy)5. | Low to Moderate (buffer management & filtering)5. | Moderate (can plateau if sampling fails to find new paths)5. | Zero without initial hits; High post-bootstrap5. |
| **S-GRPO / CGI Trajectory Injection** | GRPO combined with conditional ground-truth injection6. | High efficiency (rescues 100% of failed batches)6. | High (bridges exploration gaps via reference anchoring)6. | Low (plug-and-play extension to GRPO)6. | Low (maintains continuous policy variance)6. | High (guarantees gradient updates on all inputs)6. |
| **DreamCoder / Symbolic Search (MDL)** | Constraint enumeration \+ wake-sleep library learning13. | Extremely High CPU (combinatorial symbolic search)13. | High within shallow depth limits; low for long code13. | High (requires dedicated symbolic solver infrastructure)13. | Very Low (symbolic search guarantees structural diversity)13. | High for short algorithmic routines1. |

## **Reward Shaping and Credit Assignment Taxonomy**

In low-level assembly synthesis, the credit assignment problem is amplified by the sequential nature of stack operations24. An error in a single token early in the trajectory (e.g., pushing an incorrect constant onto the data stack) corrupts all downstream calculations, rendering terminal binary evaluation completely uninformative regarding which sub-routine was correct4. Structuring intermediate or auxiliary reward signals is therefore critical during early exploration16.  
The structural evaluation pipeline processes generated candidates through multi-tiered verification layers. At the initial structural layer, candidate token sequences undergo syntactic parsing to check WebAssembly module validity26. Sequences passing syntax validation progress to dynamic sandboxed execution, where intermediate runtime states and output sequences are evaluated against target ground-truth data5. Failed executions feed compiler and execution error metrics back to the policy, while successful executions provide prefix matching, numerical distance, and trace alignment feedback4.

### **Compiler and Parser Feedback Signals**

The first tier of dense reward shaping validates the structural integrity of the generated bytecode prior to full execution25:

> * **Compilation and Validation Reward ($R\_{\\text{comp}}$)**: Evaluates whether the generated text conforms to the WebAssembly binary specification26.

$$R\_{\\text{comp}}(y) \= \\begin{cases} \-\\gamma\_{\\text{syntax}} & \\text{if syntax / type-check fails} \\\\ \+\\gamma\_{\\text{valid}} & \\text{if WAT module compiles successfully} \\end{cases}$$  
This prevents the model from wasting policy capacity on syntactically invalid strings, pruning invalid token paths early in the generation sequence25.

> * **Abstract Syntax Tree (AST) Alignment ($R\_{\\text{AST}}$)**: Measures the structural similarity between the generated program's AST and target reference ASTs (when available) using tree-edit distance or schema matching metrics25.

### **Execution-Distance and Prefix Metrics**

When bytecode compiles and executes without trapping, continuous distance metrics quantify functional proximity to the target integer sequence $Y \= \[y\_0, y\_1, \\dots, y\_{N-1}\]$4:

> * **Prefix Match Length ($R\_{\\text{prefix}}$)**: Evaluates the longest continuous prefix of outputs that match the ground truth exactly4:

$$R\_{\\text{prefix}}(P, Y) \= \\frac{1}{N} \\max \\left\\{ k \\in \\{0, \\dots, N\\} \\;\\middle\\vert{}\\; P(n) \= y\_n \\quad \\forall n \< k \\right\\}$$  
This signal provides a smooth gradient leading toward complete sequence correctness, rewarding programs that compute initial sequence terms correctly even if downstream terms diverge due to overflow or missing loop steps4.

> * **Normalized Numerical Distance ($R\_{\\text{dist}}$)**: Employs logarithmic pointwise errors to evaluate numerical proximity across all evaluated sequence positions:

$$R\_{\\text{dist}}(P, Y) \= 1 \- \\frac{1}{N} \\sum\_{n=0}^{N-1} \\tanh \\left( \\alpha \\cdot \\left\\vert{} P(n) \- y\_n \\right\\vert{} \\right)$$  
This metric provides continuous optimization gradients for continuous function fitting, though it requires strict clipping to handle extreme numerical divergence.

### **Step-Level Semantics and Parsimony Penalties**

To resolve long-horizon token credit assignment, modern methods align token representations directly with execution semantics16:

> * **Intermediate Execution Trace Alignment ($R\_{\\text{trace}}$)**: Evaluates the correctness of intermediate stack states16. Frameworks like CodeRL+ monitor variable values and push/pop operations at each step $t$, awarding local step rewards when intermediate data stack values mirror expected execution traces16.  
> * **Minimum Description Length / Parsimony Regularization ($R\_{\\text{MDL}}$)**: Penalizes inefficient or padded bytecode (e.g., redundant nop instructions, unused stack allocations, or redundant push/pop operations)13:

$$R\_{\\text{MDL}}(P) \= R\_{\\text{exec}}(P) \- \\lambda \\cdot \\vert{}P\\vert{}\_{\\text{tokens}}$$  
This penalizes bloated, non-canonical programs and forces the policy to discover minimal programmatic representations13.

## **Empirical Pitfalls and Optimization Failures**

While dense shaped rewards prevent initial advantage collapse, transitioning a policy from surrogate reward structures to strict binary execution correctness introduces severe optimization pathologies7.

### **Degenerate Polynomial Overfitting and Reward Hacking**

The most prominent failure mode when utilizing dense numerical or prefix-based rewards is reward hacking via degenerate function fitting9. When evaluating a sequence across a finite context window of $N$ terms (e.g., $N \= 20$), the search space contains infinitely many high-degree polynomials or memorized conditional lookups that perfectly reproduce the target outputs $y\_0, \\dots, y\_{N-1}$ without capturing the underlying inductive algorithm.  
Under continuous distance rewards, the policy frequently discovers shortcuts:

> * **Lookup Table Memorization**: Generating explicit conditional trees (if n \== 0 then return 1; else if n \== 1 then return 2...) rather than generalized iterative loops.  
> * **Polynomial Interpolation**: Synthesizing high-order linear arithmetic expressions that fit initial sequence values but fail when evaluated on unseen extended terms $n \\ge N$.

When the evaluation is subsequently switched to strict binary evaluation across hidden extended test terms ($n \= N \\dots N \+ M$), the pass rate collapses to zero.

### **Advantage Collapse and the ACR Diagnostic Metric**

In group-relative policy updates (e.g., GRPO), advantage collapse remains a constant threat whenever prompt difficulties exceed current model capacity7. The Advantage Collapse Rate (ACR) quantifies the proportion of training batches within a window $B$ that yield completely zero gradients due to homogeneous rollout outcomes7:

$$\\text{ACR} \= \\frac{1}{B} \\sum\_{b=1}^{B} \\mathbb{I} \\left( \\sigma\_{\\mathbf{r}\_b}^2 \== 0 \\right)$$  
Empirical investigations demonstrate that early-stage ACR values exceeding $0.40$ (i.e., over 40% of rollout groups contributing zero gradient) correlate directly with permanent policy plateauing7. When ACR approaches $1.0$, the policy enters an exploration starvation regime where no gradient updates occur7. Intervention thresholds (typically set at $\\text{ACR} \\ge 0.30$) are required to trigger adaptive mechanisms such as S-GRPO trajectory injection or AVSPO virtual sample insertion, pulling the training trajectory back into a healthy exploitation regime6. Uncorrected high ACR causes the policy to drift into unconstrained entropy degradation or local optima7.

### **Transition Shock in Curriculum Decay**

Abruptly shifting from dense shaped rewards ($R\_{\\text{dense}}$) to sparse binary rewards ($R\_{\\text{sparse}}$) introduces severe transition shock9. If the policy has optimized its parameters to maximize intermediate AST matching or prefix alignment, the loss surface changes fundamentally when those dense surrogate signals are removed9.  
When the policy gradient objective switches instantly:

$$R\_{\\text{total}} \\leftarrow R\_{\\text{sparse}}$$  
the policy gradient vectors undergo drastic directional shifts9. The policy quickly unlearns partially functional programmatic structures that yielded moderate dense rewards but failed exact binary verification, causing catastrophic forgetting of basic control-flow constructs9.

## **Recommended Hybrid Training Architecture Workflow**

To synthesize low-level WebAssembly Text (WAT) bytecode from integer sequences without encountering cold-start starvation or reward-hacking failure modes, a four-stage hybrid curriculum pipeline is recommended5.  
The curriculum progresses sequentially through structured learning phases. Stage 1 establishes fundamental syntax grounding via Supervised Fine-Tuning on synthetic execution pairs10. Stage 2 executes offline symbolic search (DreamCoder MDL enumeration) to populate an elite seed replay buffer with reference solutions5. Stage 3 initiates online exploration via Supervised Group Relative Policy Optimization (S-GRPO), leveraging Conditional Ground-Truth Trajectory Injection (CGI) and AVSPO to eliminate zero-advantage failure groups6. Stage 4 transitions the policy to pure Verifiable Rewards (RLVR) using cosine-decayed shaping terms and teacher KL divergence constraints to enforce exact functional correctness4.

### **Stage 1: Synthetic Forward Corpus SFT Warmup**

Training begins with an offline Supervised Fine-Tuning phase to initialize the Transformer encoder-decoder policy $\\pi\_\\theta$4. A synthetic program generator samples syntactically valid WAT programs containing core stack operations (i32.add, i32.sub, i32.mul, i32.div\_s, local.get, local.set, loop, block, br\_if)2. Programs are executed across $n \\in \\{0, \\dots, N-1\\}$ to generate pairs $(Y, P)$10. The model optimizes standard next-token prediction loss over $P$ conditioned on sequence encoding $Y$, establishing baseline syntax compliance and preventing execution trap generation17.

### **Stage 2: Symbolic Search-Guided Buffer Population**

Before initiating online policy gradients, target integer sequences in the training set are processed through an enumerative symbolic search engine (or stack-based genetic search engine) bounded by a strict execution timeout13. Found programs are filtered using Minimum Description Length (MDL) criteria to select the shortest valid bytecode representation for each sequence13. The resulting programs populate an Elite Replay Buffer $\\mathcal{D}\_{\\text{elite}}$, ensuring that baseline reference solutions exist for a substantial subset of training tasks5.

### **Stage 3: Guided Group Relative Policy Optimization (S-GRPO \+ AVSPO)**

Online reinforcement learning employs Group Relative Policy Optimization augmented with Conditional Ground-Truth Trajectory Injection (S-GRPO)6. For each problem prompt $q\_j$, the model samples $G$ completions $\\{y\_1, \\dots, y\_G\\}$6.

> * **Failure Group Injection**: If all $G$ samples fail binary verification ($r\_i \= \-1 \\ \\forall i$), the system fetches a reference program $y^\*\_{\\text{gt}}$ from $\\mathcal{D}\_{\\text{elite}}$ and injects it into the group6. Advantage normalization is performed over the mixed group, forcing an imitation policy gradient step on $y^\*\_{\\text{gt}}$ while penalizing the failed samples6.  
> * **Real-time Diagnostic Monitoring**: The Advantage Collapse Rate (ACR) is computed across sliding batch windows7. If ACR exceeds $\\tau \= 0.30$, Adaptive Virtual Sample Policy Optimization (AVSPO) activates, injecting virtual positive anchor rewards to ensure non-zero gradient flow across hard prompts7.

### **Stage 4: Verifiable RLVR with Cosine Decay**

In the final phase, the policy transitions to pure Reinforcement Learning from Verifiable Rewards (RLVR)4. To avoid transition shock, dense auxiliary rewards (compiler validity, prefix distance, and teacher KL-divergence constraints) are gradually decayed using a cosine schedule over $S$ training steps9:

$$R\_{\\text{total}}(P) \= R\_{\\text{exact}}(P) \+ \\cos \\left( \\frac{\\pi \\cdot s}{2S} \\right) \\cdot \\left\[ \\beta\_1 R\_{\\text{comp}}(P) \+ \\beta\_2 R\_{\\text{prefix}}(P) \\right\]$$  
This smooth annealing forces the policy to transition from relying on dense heuristic guidance to optimizing exact binary execution correctness over extended evaluation windows9.

| Training Phase | Optimization Objective & Loss Formulation | Active Reward Signal | Primary Failure Mode Addressed | Exit / Transition Criteria |
| :---- | :---- | :---- | :---- | :---- |
| **Phase 1: Synthetic SFT Warmup** | MLE Cross-Entropy: $\\mathcal{L}\_{\\text{SFT}}(\\theta) \= \-\\log \\pi\_\\theta(P \\mid Y)$17. | N/A (Supervised teacher forcing)17. | High token entropy; ungrammatical WAT generation10. | WAT compilation pass rate $\> 95\\%$ on random prompts25. |
| **Phase 2: Symbolic Search Population** | MDL Optimization: $\\arg\\min\_P \[ \-\\log P(P) \+ \\text{Cost}\_{\\text{exec}} \]$13. | Binary ground-truth execution correctness13. | Absence of initial training demonstrations5. | $\\ge 30\\%$ of training set prompts paired with valid seed $P^\*$5. |
| **Phase 3: S-GRPO \+ CGI Exploration** | Group Policy Gradient with Injected Trajectory Advantages6. | Binary Reward \+ CGI Reference Reward ($r\_{\\text{gt}} \= \+1$)6. | Zero-Advantage Collapse ($\\text{ACR} \\to 1.0$)6. | $\\text{ACR} \< 0.15$ without trajectory injection assistance7. |
| **Phase 4: Annealed RLVR Fine-Tuning** | KL-constrained Policy Gradient with Cosine-Decayed Shaping4. | Exact Binary Reward $R\_{\\text{exact}} \\in \\{-1, \+1\\}$4. | Degenerate polynomial overfitting & reward hacking9. | Convergence of pass@1 accuracy on hidden test sets16. |

## **Synthesis of Academic Literature and Key Benchmarks**

The literature on neuro-symbolic synthesis, reinforcement learning from execution rewards, and cold-start exploration spans foundational machine learning and programming language venues including NeurIPS, ICML, ICLR, and POPL5.

| Paradigm / Framework | Key Authors & Citation | Primary Venue | Benchmark / Domain | Key Technical Innovation |
| :---- | :---- | :---- | :---- | :---- |
| **DreamCoder** | Ellis et al.13 | NeurIPS / Nature | List processing, Graphics, Regex | Wake-sleep Bayesian program learning with MDL library abstraction13. |
| **CodeRL** | Le et al.18 | NeurIPS / ICLR | APPS, MBPP | Actor-critic token RL with execution unit test feedback & critic sampling18. |
| **CodeRL+** | Liu et al.16 | arXiv 2025 | HumanEval, LeetCode, LiveCodeBench | Execution semantics alignment via intermediate variable trajectory inference16. |
| **S-GRPO** | Zhang et al.6 | arXiv 2026 | Visual-math & Sparse preference tasks | Conditional Ground-Truth Trajectory Injection (CGI) to eliminate zero-advantage collapse6. |
| **AVSPO** | Xu et al.7 | arXiv 2026 | Mathematical Reasoning (AMC, AIME) | Real-time Advantage Collapse Rate (ACR) metric & virtual sample injection7. |
| **MAPO** | Liang et al.5 | NeurIPS | WikiTableQuestions | Memory-augmented policy gradient with memory weight clipping for cold start5. |
| **RobustFill** | Devlin et al.10 | ICML | String Manipulation | Synthetic forward-generation warmup for inductive program synthesis10. |
| **OEIS Synthesis** | Guo et al.1 | arXiv / AITP | On-Line Encyclopedia of Integer Sequences | Self-learning search guided by neural policy over stack/loop primitives1. |
| **Stack-Based GP / PushGP** | Spector et al.15 | IEEE TEC / GPEM | Symbolic Regression & Program Synthesis | Stack-based genetic programming with autonomous data stack manipulation15. |

Synthesizing low-level WebAssembly Text bytecode from integer sequences requires unifying parametric autoregressive transformers with explicit execution semantics2. As demonstrated across recent advancements in verifiable reinforcement learning, resolving the cold-start sparse-reward problem cannot rely on policy-gradient updates alone6. By combining forward synthetic pretraining, search-based demonstration buffers, conditional trajectory injection, and cosine-decayed execution reward shaping, neuro-symbolic synthesizers achieve stable gradient flow and robust programmatic generalization5.

#### **Works cited**

> 1. Learning Program Synthesis for Integer Sequences from Scratch, [https://arxiv.org/abs/2202.11908](https://arxiv.org/abs/2202.11908)  
> 2. Improvements in Program Synthesis for Integer Sequences, [http://aitp-conference.org/2023/abstract/AITP\_2023\_paper\_8.pdf](http://aitp-conference.org/2023/abstract/AITP_2023_paper_8.pdf)  
> 3. Revealing Performance Issues in Server-side WebAssembly ... \- arXiv, [https://arxiv.org/html/2309.12167v1](https://arxiv.org/html/2309.12167v1)  
> 4. A Technical Survey of Reinforcement Learning Techniques for, [https://arxiv.org/html/2507.04136v2](https://arxiv.org/html/2507.04136v2)  
> 5. Memory Augmented Policy Optimization for Program Synthesis and, [https://arxiv.org/pdf/1807.02322](https://arxiv.org/pdf/1807.02322)  
> 6. S-GRPO: Unified Post-Training for Large Vision-Language Models, [https://arxiv.org/html/2604.16557v1](https://arxiv.org/html/2604.16557v1)  
> 7. Advantage Collapse in Group Relative Policy Optimization \- arXiv, [https://arxiv.org/html/2605.21125v1](https://arxiv.org/html/2605.21125v1)  
> 8. GHPO: Adaptive Guidance for Stable and Efficient LLM ... \- arXiv, [https://arxiv.org/html/2507.10628v1](https://arxiv.org/html/2507.10628v1)  
> 9. Prior Injection for Sparse-Reward RL in Vision–Language Math, [https://arxiv.org/html/2608.21811v1](https://arxiv.org/html/2608.21811v1)  
> 10. arXiv:2204.03758v1 \[cs.LG\] 7 Apr 2022, [https://arxiv.org/pdf/2204.03758](https://arxiv.org/pdf/2204.03758)  
> 11. Case2Code: Scalable Synthetic Data for Code Generation \- arXiv, [https://arxiv.org/html/2407.12504v2](https://arxiv.org/html/2407.12504v2)  
> 12. S-GRPO: Unified Post-Training for Large Vision-Language Models, [https://arxiv.org/abs/2604.16557](https://arxiv.org/abs/2604.16557)  
> 13. Neural networks for abstraction and reasoning \- PMC \- NIH, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11561310/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11561310/)  
> 14. growing generalizable, interpretable knowledge with wake–sleep, [https://www.researchgate.net/publication/371306616\_DreamCoder\_growing\_generalizable\_interpretable\_knowledge\_with\_wake-sleep\_Bayesian\_program\_learning](https://www.researchgate.net/publication/371306616_DreamCoder_growing_generalizable_interpretable_knowledge_with_wake-sleep_Bayesian_program_learning)  
> 15. Recent Developments in Program Synthesis with Evolutionary, [https://arxiv.org/abs/2108.12227](https://arxiv.org/abs/2108.12227)  
> 16. CodeRL+: Improving Code Generation via Reinforcement with, [https://arxiv.org/html/2510.18471v2](https://arxiv.org/html/2510.18471v2)  
> 17. The Landscape of Agentic Reinforcement Learning for LLMs: A Survey, [https://openreview.net/pdf/e9289574e24a3a2ff62a3af86cec1fa2f189ce54.pdf](https://openreview.net/pdf/e9289574e24a3a2ff62a3af86cec1fa2f189ce54.pdf)  
> 18. CodeRL: Mastering Code Generation through Pretrained Models, [https://arxiv.org/abs/2207.01780](https://arxiv.org/abs/2207.01780)  
> 19. Value-Based Deep Reinforcement Learning for Program Synthesis, [https://arxiv.org/pdf/2310.03173](https://arxiv.org/pdf/2310.03173)  
> 20. Value-Based Deep Reinforcement Learning for Program Synthesis, [https://arxiv.org/html/2310.03173v2](https://arxiv.org/html/2310.03173v2)  
> 21. SYNTHESIZING PROGRAMMATIC REINFORCEMENT LEARNING, [https://proceedings.iclr.cc/paper\_files/paper/2025/file/dcf887f2bfe2776584e3bce80ed206ef-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/dcf887f2bfe2776584e3bce80ed206ef-Paper-Conference.pdf)  
> 22. Transductively Informed Inductive Program Synthesis \- arXiv, [https://arxiv.org/html/2505.14744v1](https://arxiv.org/html/2505.14744v1)  
> 23. B \- \-Coder: Value-Based Deep Reinforcement Learning for Program, [https://openreview.net/forum?id=fLf589bx1f](https://openreview.net/forum?id=fLf589bx1f)  
> 24. PROGRAMMING WITH A DIFFERENTIABLE FORTH IN \- OpenReview, [https://openreview.net/pdf?id=SJAM\_pVte](https://openreview.net/pdf?id=SJAM_pVte)  
> 25. Execution-based Code Generation using Deep Reinforce, [https://creddy.net/papers/TMLR23a.pdf](https://creddy.net/papers/TMLR23a.pdf)  
> 26. Distinguishability-guided Test Program Generation for ... \- arXiv, [https://arxiv.org/html/2412.20100v1](https://arxiv.org/html/2412.20100v1)  
> 27. Execution-based Code Generation using Deep Reinforcement, [https://openreview.net/forum?id=0XBuaxqEcG](https://openreview.net/forum?id=0XBuaxqEcG)  
> 28. A First-Principles Derivation of LLM Policy Optimization \- arXiv, [https://arxiv.org/html/2606.16733v1](https://arxiv.org/html/2606.16733v1)  
> 29. GP and LLMs for Program Synthesis: No Clear Winners \- arXiv, [https://arxiv.org/pdf/2508.03966](https://arxiv.org/pdf/2508.03966)