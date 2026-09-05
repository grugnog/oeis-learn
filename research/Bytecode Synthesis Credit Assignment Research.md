# **Step-Level Credit Assignment and Semantic Execution Trace Alignment for Low-Level Bytecode Synthesis**

## **Theoretical Foundations and Failure Modes of Coarse Reinforcement Learning in Bytecode Synthesis**

Neural program synthesis has increasingly shifted from high-level, human-readable programming languages toward low-level, stack-based bytecodes, such as WebAssembly Text (WAT) S-expressions, Java Virtual Machine (JVM) bytecodes, and virtual machine postfix instruction streams1. Stack-based bytecodes present an attractive target for automated code generation due to their compact instruction sets, lack osee the codebase and f ambiguous syntax trees, and direct compilation to sandboxed execution environments1. However, synthesizing low-level bytecode autoregressively using Large Language Models (LLMs) and policy gradient reinforcement learning introduces severe optimization bottlenecks that do not exist in high-level language synthesis1.  
Unlike high-level languages where program structure is explicitly delineated by abstract syntax trees (ASTs), block scopes, and named identifiers, a stack-based bytecode machine operates via an implicit, dynamic execution state1. At any execution step $t$, the state of a stack machine can be formally defined as a tuple:

$$\\Omega\_t \= (\\text{PC}\_t, \\Sigma\_t, \\Gamma\_t, M\_t)$$  
where $\\text{PC}\_t \\in \\mathbb{N}$ denotes the program counter, $\\Sigma\_t \= \[v\_1, v\_2, \\dots, v\_{d\_t}\]^\\top$ represents the operand stack of depth $d\_t$, $\\Gamma\_t: \\mathbb{N} \\to \\mathcal{V}$ maps local variable indices to values in value domain $\\mathcal{V}$, and $M\_t$ represents linear memory1. Program execution proceeds via a deterministic transition function $\\mathcal{E}: \\Omega\_t \\times a\_t \\to \\Omega\_{t+1}$, where $a\_t$ is the bytecode instruction token emitted by the policy model $\\pi\_\\theta(a\_t \\mid s\_t)$ at generation step $t$1.  
The fundamental difficulty in training neural policies for stack bytecode synthesis via standard Reinforcement Learning with Verifiable Rewards (RLVR)—such as REINFORCE, Proximal Policy Optimization (PPO), or Group Relative Policy Optimization (GRPO)—lies in the extreme credit assignment fragility inherent to postfix stack manipulation4. In standard GRPO, a group of $G$ program completions $\\{y\_i\\}\_{i=1}^G$ is sampled for a given specification $x$4. Each generated program $y\_i \= (a\_{i,1}, a\_{i,2}, \\dots, a\_{i,T\_i})$ is executed against a suite of test cases to yield a binary or continuous terminal outcome reward $R(y\_i) \\in \[-1, 1\]$1. The policy optimization objective is formulated as:

$$\\mathcal{L}\_{\\text{GRPO}}(\\theta) \= \-\\frac{1}{G} \\sum\_{i=1}^G \\sum\_{t=1}^{T\_i} \\min \\left( \\frac{\\pi\_\\theta(a\_{i,t} \\mid s\_{i,t})}{\\pi\_{\\text{old}}(a\_{i,t} \\mid s\_{i,t})} \\hat{A}\_i, \\; \\text{clip}\\left(\\frac{\\pi\_\\theta(a\_{i,t} \\mid s\_{i,t})}{\\pi\_{\\text{old}}(a\_{i,t} \\mid s\_{i,t})}, 1-\\epsilon, 1+\\epsilon\\right) \\hat{A}\_i \\right)$$  
where the sequence-level advantage $\\hat{A}\_i$ is computed by normalizing outcome rewards across the sampled group4:

$$\\hat{A}\_i \= \\frac{R(y\_i) \- \\frac{1}{G}\\sum\_{j=1}^G R(y\_j)}{\\sqrt{\\frac{1}{G}\\sum\_{j=1}^G \\left(R(y\_j) \- \\frac{1}{G}\\sum\_{k=1}^G R(y\_k)\\right)^2 \+ \\varepsilon}}$$  
This formulation broadcasts a static scalar return $\\hat{A}\_i$ uniformly across every token $t \\in \[1, T\_i\]$ in the generated sequence4. In low-level stack synthesis, this spatial and temporal broadcasting induces two severe failure modes:  
The first structural vulnerability is the catastrophic failure cascade caused by single token corruption4. Stack instructions depend strictly on the exact depth and type composition of $\\Sigma\_t$1. If the policy emits a single erroneous instruction early in the sequence—such as pushing an incorrect immediate constant (i64.const 0 instead of i64.const 1), swapping operand ordering (i32.sub vs binary swap), or omitting a local retrieval (local.get $n)—the state transition $\\mathcal{E}(\\Omega\_t, a\_t)$ yields a corrupted state $\\Omega\_{t+1}$1. Every subsequent instruction $a\_{t+k}$ operates on an invalid stack frame, triggering execution traps such as stack underflow or invalid memory access, or producing completely incorrect arithmetic results4.  
The second failure mode is severe policy degradation caused by credit smear4. When a program fails ($R(y\_i) \= \-1$), standard policy gradients assign a harsh negative advantage $\\hat{A}\_i \< 0$ uniformly to all $T\_i$ tokens4. Consequently, valid module headers (such as (module (func (param i32)))), functional variable initializations, and correct control-flow prefixes are heavily penalized alongside the single faulty token4. This uniform penalty destroys the policy's structural fluency, destabilizes training, increases policy gradient variance, and suppresses exploration of valid algorithmic prefixes4.  
Addressing this fundamental mismatch between uniform sequence-level rewards and fine-grained, localized execution errors requires moving from outcome-level reinforcement learning to token-level credit assignment, execution trace alignment, and soft semantic execution semantics1.

## **Execution-Grounded Credit Assignment and Instruction Attribution**

To eliminate credit smear, recent breakthroughs in reinforcement learning for code generation have focused on Execution-Grounded Credit Assignment (EGCA), isolating exact execution failure boundaries and mapping them back to token-level advantage masks4. Rather than treating the generated program as an indivisible string, EGCA frameworks utilize execution traces, compiler diagnostics, and unit test outputs to localize semantic divergence4.

### **Line and Token Attribution Frameworks**

Several distinct architectural approaches have been developed to implement fine-grained attribution in code generation, addressing the limitations of uniform sequence-level reward propagation through specialized credit masking and variable state alignment1.  
The StepCoder framework addresses exploration complexity and unexecuted token credit assignment through its Curriculum of Code Completion Subtasks (CCCS) and Fine-Grained Optimization (FGO) mechanisms7. FGO tracks code coverage during unit test execution7. Tokens corresponding to instruction branches that were never executed during test runs are dynamically masked out from the policy gradient update7. For an executed token sequence $\\tau$ with execution mask $M\_t \\in \\{0, 1\\}$, the FGO loss updates only executed instructions7:

$$\\mathcal{L}\_{\\text{FGO}}(\\theta) \= \-\\sum\_{t=1}^T M\_t \\cdot \\log \\pi\_\\theta(a\_t \\mid s\_t) \\cdot \\hat{A}$$  
While FGO prevents unexecuted code from receiving spurious credit or blame, it provides no fine-grained disambiguation among tokens when programs execute fully to completion but fail logically4.  
The CodeRL+ architecture augments standard RLVR by introducing an auxiliary variable-level execution trajectory learning objective1. When a sampled program rollout $p\_{\\text{fail}}$ fails exploration, CodeRL+ extracts the ground-truth variable propagation trace $F\_{p\_{\\text{fail}}}(x)\[v\_k\]$ across execution steps1. The model is trained to jointly generate code and predict the intermediate and final states of runtime variables $\\hat{v}\_k^{\\text{final}}$1. The joint optimization objective is formulated as1:

$$\\mathcal{L}\_{\\text{CodeRL+}}(\\theta) \= \\mathbb{E}\_{q \\sim \\mathcal{B}\_{\\text{code}}} \[r(\\theta) \\cdot A\_{\\text{gen}}\] \+ \\mathbb{E}\_{q' \\sim \\mathcal{B}\_{\\text{align}}} \[r'(\\theta) \\cdot A\_{\\text{sem}}\]$$  
where $A\_{\\text{sem}}$ evaluates the precision of predicting variable state transitions $R\_{\\text{sem}} \= \\frac{1}{\\vert{}V\\vert{}} \\sum\_{v\_k \\in V} \\mathbb{I}\[\\hat{v}\_k^{\\text{final}} \= v\_k^{\\text{final}, \*}\]$, aligning the model's internal representation with runtime execution semantics1.  
The StepCodeReasoner framework converts black-box code generation into an observable execution modeling process by automatically inserting structured execution-trace anchors (\<print\> statements) at critical control-flow and variable update points8. The model is trained to interleave reasoning blocks with explicit state predictions8. Credit assignment is performed via Bi-Level GRPO, which computes relative advantages across alternative trajectories and rewards intermediate state correctness proportional to downstream task success8.  
The Execution-Grounded Credit Assignment (EGCA) method for GRPO explicitly localizes the earliest point of semantic divergence between a generated program's execution trace and a reference execution trace, concentrating gradient updates precisely on the causal token window4.

### **Mathematical Formulation of Divergence Localization and Advantage Masking**

In the EGCA paradigm, generated bytecodes are categorized into priority-ordered failure modes using deterministic gates4:

$$m(y) \= \\begin{cases} \\text{SYNTAX} & \\text{if } y \\text{ raises compilation, parse, or type errors,} \\\\ \\text{CONSTRAINT} & \\text{if } y \\text{ violates structural/algorithmic invariants } I\_C(y) \= 0, \\\\ \\text{CORRECT} & \\text{if } \\hat{R}(y) \= 1 \\text{ (passes all test cases)}, \\\\ \\text{LOGIC} & \\text{otherwise (compiles and executes, but fails test cases).} \\end{cases}$$  
When a program falls into the LOGIC failure mode, divergence localization is triggered4. Let $d$ be the first failing test case input4. The program execution is divided into $K$ execution boundaries $B(y) \= (b\_1, b\_2, \\dots, b\_K)$, where each boundary $b\_k$ maps directly to a span of bytecode instruction tokens $T\_k \\subset \\{1, \\dots, T\\}$4. Executing the candidate program $y$ and an offline canonical reference program $y\_{\\text{ref}}$ under input $d$ generates paired execution state traces4:

$$\\tau(y, d) \= (S\_1, S\_2, \\dots, S\_K), \\quad \\tau(y\_{\\text{ref}}, d) \= (S\_1^{\\text{ref}}, S\_2^{\\text{ref}}, \\dots, S\_K^{\\text{ref}})$$  
The index of the earliest semantic divergence boundary $k^\*$ is identified as4:

$$k^\* \= \\min \\{ k \\in \\{1, \\dots, K\\} : S\_k \\neq S\_k^{\\text{ref}} \\}$$  
The corresponding token span $T\_{k^\*}$ represents the causal error window where execution diverged4. Rather than applying the sequence return $A\_i$ to all tokens, EGCA constructs a localized token-level advantage operator $a\_{i,t}$4:

$$a\_{i,t} \= \\begin{cases} \\frac{A\_i}{T\_i} & \\text{if } m(y\_i) \\in \\{\\text{CORRECT}, \\text{CONSTRAINT}\\}, \\\\ \\frac{A\_i}{\\vert{}T\_{\\text{err}}\\vert{}} \\mathbf{1}\[t \\in T\_{\\text{err}}\] & \\text{if } m(y\_i) \= \\text{SYNTAX}, \\\\ \\frac{A\_i}{\\vert{}T\_{k^\*}\\vert{}} \\mathbf{1}\[t \\in T\_{k^\*}\] & \\text{if } m(y\_i) \= \\text{LOGIC}. \\end{cases}$$  
where $T\_{\\text{err}}$ is the token span extracted from compiler error diagnostics4. Crucially, for LOGIC and SYNTAX failures, all tokens generated *after* the causal error span ($t \> \\max T\_{k^\*}$) are masked out ($\\mathbf{1}\[t \\in T\_{k^\*}\] \= 0$), preventing downstream boilerplate from receiving negative reinforcement4. Furthermore, the operator maintains exact total advantage normalization $\\sum\_{t=1}^{T\_i} a\_{i,t} \= A\_i$, ensuring that the global gradient scale remains stable while gradient mass is concentrated entirely on the causal token window4.

| Attribution Framework | Granularity of Credit | Execution Instrumentation Method | Handling of Unexecuted / Downstream Tokens | Reliance on Reference Solutions |
| :---- | :---- | :---- | :---- | :---- |
| **StepCoder (FGO)** \[cite: 7, 9\] | Line / Block level | Compiler code coverage tracing | Unexecuted tokens are masked ($M\_t \= 0$)7 | No reference solution required during RL updates7 |
| **CodeRL+** \[cite: 1, 10, 11\] | Variable state level | Intermediate trajectory variable extraction | Evaluates variable transitions across rollouts1 | Uses failed rollouts for trajectory alignment1 |
| **StepCodeReasoner** \[cite: 8, 12\] | Anchor step level | Trace anchor insertion (\<print\> statements) | Evaluates step correctness via Bi-Level GRPO8 | Relies on ground-truth execution anchor outputs8 |
| **EGCA (GRPO)** \[cite: 4\] | Divergent Token Span | Pairwise state trace alignment $\\tau(y, d)$ vs $\\tau(y\_{\\text{ref}}, d)$ | Downstream tokens ($t \> k^\*$) are masked to 04 | Requires reference execution trace for divergence matching4 |

## **Process Reward Models and Value Functions over Execution States**

While non-parametric trace comparison heuristics isolate divergence boundaries effectively when reference solutions exist, synthesizing complex, novel bytecodes requires parametric verifiers capable of assigning localized rewards to unseen intermediate execution states3. Process Reward Models (PRMs) and state-space value functions fulfill this requirement by evaluating intermediate program states $\\Omega\_t \= (\\text{PC}\_t, \\Sigma\_t, \\Gamma\_t, M\_t)$ directly3.

### **Parametric Process Verifiers vs. Non-Parametric Trace Comparison Heuristics**

Process supervision can be established via two primary paradigms, each presenting trade-offs between generalization capabilities and execution stability3.  
In parametric process reward models such as CodePRM, a dense critic network $R\_\\phi(\\Omega\_t, a\_t)$ is trained via step-supervised regression or Monte Carlo rollout estimation to predict the probability that the partial sequence generated up to step $t$ can be completed into a functionally correct program3. In stack bytecode synthesis, $R\_\\phi$ takes the linearized sequence of emitted instructions and the current stack frame snapshot $\\Sigma\_t$ as input, outputting a step advantage $A\_t^{\\text{PRM}} \= R\_\\phi(\\Omega\_t, a\_t) \- V\_\\phi(\\Omega\_t)$3. While highly sample-efficient and capable of generalizing to novel instruction sequences, parametric critics are prone to value estimation drift and distribution mismatch when evaluating out-of-distribution code constructs3.  
In contrast, non-parametric trace comparison heuristics such as EGCA bypass parametric value training entirely by utilizing deterministic execution trace diffing4. These heuristics compute localized advantages by comparing runtime stack states directly against canonical reference traces4. Non-parametric heuristics eliminate value estimation drift and distribution mismatch entirely, but their applicability is limited to settings where canonical reference execution traces are available4.

### **Bi-Level GRPO and Intermediate Advantage Formulation**

To integrate step-level process supervision into critic-free algorithms without incurring the severe policy divergence associated with noisy process reward signals, Bi-Level GRPO decomposes credit assignment into two distinct advantage layers8.  
The inter-trajectory advantage performs cross-trajectory relative comparisons across a sampled group of $G$ program rollouts for each execution step or anchor $i$, filtering out baseline task difficulty8:

$$\\hat{A}\_{i,g}^{\\text{group}} \= \\frac{r\_{i,g} \- \\frac{1}{G}\\sum\_{g'=1}^G r\_{i,g'}}{\\sqrt{\\frac{1}{G}\\sum\_{g'=1}^G \\left(r\_{i,g'} \- \\frac{1}{G}\\sum\_{k=1}^G r\_{i,k}\\right)^2 \+ \\varepsilon}}$$  
where $r\_{i,g} \\in \\{0, 1\\}$ indicates whether trajectory $g$ maintained valid stack state invariants and intermediate results at anchor step $i$8.  
To prevent reward hacking—where a model generates locally valid stack operations that ultimately lead to dead-end states—the intra-trajectory advantage scales intermediate step credit by the correctness of all subsequent steps8:

$$\\hat{A}\_{i,g}^{\\text{intra}} \= r\_{i,g} \\cdot \\left( 1 \+ \\frac{1}{n \- i} \\sum\_{j \= i+1}^n r\_{j,g} \\right)$$  
where $n$ is the total number of execution anchor steps in the sequence8. The intra-trajectory term enforces two strict properties: if an intermediate step is incorrect ($r\_{i,g} \= 0$), its shaped advantage drops to zero regardless of downstream luck; and an intermediate step receives maximal credit only if it is locally correct ($r\_{i,g} \= 1$) and successfully enables downstream execution steps ($\\frac{1}{n-i}\\sum\_{j=i+1}^n r\_{j,g} \\to 1$)8.

| System Component | Parametric PRM Critic (e.g., CodePRM) | Non-Parametric Trace Heuristics (e.g., EGCA) | Bi-Level Advantage System (StepCodeReasoner) |
| :---- | :---- | :---- | :---- |
| **Sample Efficiency** | High; generalizes across novel states via continuous representations3. | Medium; restricted to paths that align with reference executions4. | Very High; combines relative group sampling with step-shaping8. |
| **Computational Complexity** | High; requires training and inference over a separate critic model3. | Low; requires only deterministic trace comparisons4. | Moderate; computes relative stats directly over sampled rollouts8. |
| **Distribution Mismatch Vulnerability** | High; PRM critic can suffer from value estimation drift on out-of-distribution code17. | Zero; execution feedback is exact and verifiable4. | Extremely Low; verified step anchors prevent value drift8. |
| **Susceptibility to Reward Hacking** | High; policy can exploit flaws in learned reward function8. | None; binary trace matching is deterministic4. | Prevented via intra-trajectory downstream validation8. |

## **Differentiable Interpreters and Soft Execution Semantics**

An alternative to discrete policy gradient optimization is the continuous relaxation of bytecode interpreters2. Differentiable interpreters relax discrete stack operations, register transitions, and control-flow jumps into soft, continuous tensor transformations2. By making the execution environment itself differentiable, gradient backpropagation can flow directly from the final numerical execution loss through the entire execution trace down to the model's token selection weights2.

### **Continuous Relaxation of Stack Machines**

In a discrete stack machine, pushing an element increments the stack pointer $sp \\in \\mathbb{N}$ and sets $\\Sigma\[sp\] \= v$2. In a soft stack machine—such as Differentiable Forth ($\\partial 4$) or Differentiable Tree Machines (DTM)—the discrete stack pointer is relaxed into a soft probability distribution vector $\\mathbf{s}\_t \\in \\Delta^S$ over maximum stack depth $S$2:

$$\\mathbf{s}\_t \= \[p\_0, p\_1, \\dots, p\_S\]^\\top, \\quad \\sum\_{j=0}^S p\_j \= 1$$  
The operand stack is represented as a matrix $\\mathbf{V}\_t \\in \\mathbb{R}^{S \\times D}$, where each row $j$ holds a $D$-dimensional continuous representation or soft value distribution18. When the model selects an instruction from vocabulary $\\mathcal{O}$ with continuous probability vector $\\mathbf{\\pi}\_t \= \\text{softmax}(\\mathbf{z}\_t) \\in \\mathbb{R}^{\\vert{}\\mathcal{O}\\vert{}}$, the execution step updates both the soft stack pointer distribution $\\mathbf{s}\_t$ and the soft stack content matrix $\\mathbf{V}\_t$ via linear algebraic operators18.  
Let $\\mathbf{T}\_{\\text{push}}, \\mathbf{T}\_{\\text{pop}}, \\mathbf{T}\_{\\text{nop}} \\in \\mathbb{R}^{(S+1) \\times (S+1)}$ be shift matrices defining stack pointer movements18. The soft stack pointer transitions as a weighted linear combination of candidate instruction actions18:

$$\\mathbf{s}\_{t+1} \= \\sum\_{o \\in \\mathcal{O}} \\mathbf{\\pi}\_t\[o\] \\cdot \\left( \\mathbf{T}\_o \\mathbf{s}\_t \\right)$$  
Instruction values and memory updates are calculated via tensor contractions20. For an instruction selecting operands $v\_{1,t}$ and $v\_{2,t}$ from the soft stack, the resulting output representation $o\_t$ is computed as20:

$$o\_t \= \\text{einsum}(klmn, k, l, m \\to n, \\; \\mathbf{T}\_{\\text{op}}, \\; \\mathbf{\\pi}\_t, \\; v\_{1,t}, \\; v\_{2,t})$$  
The soft stack matrix is subsequently updated via continuous interpolation:

$$\\mathbf{V}\_{t+1} \= \\mathbf{V}\_t \\odot (\\mathbf{1} \- \\mathbf{s}\_{t+1} \\mathbf{r}\_t^\\top) \+ (o\_t \\otimes \\mathbf{s}\_{t+1}) \\mathbf{r}\_t^\\top$$  
where $\\mathbf{r}\_t$ designates argument write-mask vectors20.

### **Neural Compilation and Differentiable Meta-Circular Interpretation**

Recent architectures extend continuous execution beyond specialized Forth interpreters2:  
Differentiable Meta-Circular Interpretation (DMCI) compiles a self-hosting Scheme interpreter into an autograd-compliant PyTorch computational graph2. When evaluating programs containing learnable discrete or continuous parameters, reverse-mode automatic differentiation propagates gradients directly through the interpreter's control flow, environment lookups, and heap operations back to the generating policy parameters $\\theta$2.  
In neural compilation frameworks, high-level ASTs or low-level assembly statements are compiled directly into the weight matrices of recurrent update blocks, allowing backpropagation through long execution traces20.

### **Structural Limitations of Soft Stack Execution**

Despite their theoretical elegance, soft execution semantics encounter three severe scaling bottlenecks when applied to real-world bytecode synthesis18.  
First, backpropagating gradients through $T$ soft execution steps requires unrolling the continuous computational graph over time18. Matrix multiplications over soft stack distributions $\\mathbf{s}\_t$ cause exponential norm decay (vanishing gradients) or rapid overflow (exploding gradients) as $T \> 100$, severely limiting differentiable interpreters to short code sketches18.  
Second, computing soft operations requires evaluating every instruction $o \\in \\mathcal{O}$ in superposition at every step $t$20. The spatial and temporal complexity scales as $O(T \\cdot \\vert{}\\mathcal{O}\\vert{} \\cdot S^2 \\cdot D)$, making dense stack machine differentiable interpretation computationally intractable for modern instruction sets like WebAssembly ($\\vert{}\\mathcal{O}\\vert{} \> 200$, $S \> 1024$)20.  
Third, as execution progresses, continuous convex combinations over stack states blur sharp discrete operations18. A soft stack pointer spread across indices $\[2, 3, 4\]$ loses the precise integer alignment required for stack opcodes (i32.add popping exactly two elements), causing the continuous relaxation to diverge fundamentally from true discrete machine behavior18.

## **Execution-Guided Autoregressive Decoding and Search Integration**

To reconcile the exact discrete semantics of stack bytecodes with autoregressive language generation, recent systems interleave sandboxed execution directly into the decoding process3. Execution-Guided Decoding eliminates ungrammatical, stack-invalid, and logically divergent paths during inference before token commitments are made3.

### **Interleaved Sandboxed Execution Pipeline**

During autoregressive generation, the policy network emits candidate bytecode instruction tokens $a\_t \\sim \\pi\_\\theta(\\cdot \\mid s\_t)$3. Instead of sampling blindly, a lightweight, sandboxed WebAssembly or virtual machine interpreter runs in parallel with the LLM key-value (KV) cache3.  
Execution guidance operates through a tightly coupled loop between the policy generator and the sandboxed virtual machine3. At each step, candidate instruction logits produced by the policy model are first routed through a static stack masking filter3. This filter inspects the current virtual machine state $\\Omega\_t$ and assigns a logit mask of $-\\infty$ to any opcode that would trigger an immediate structural trap, such as a stack underflow3.  
The filtered token distribution is then sampled, and the chosen instruction is executed speculatively within the sandbox3. If the step yields a valid state transition, the instruction is committed to the sequence, and the corresponding Key-Value (KV) cache state is updated3. If execution results in a trap or divergence, the branch is immediately pruned, triggering backtracking within the decoding search tree8.

### **Static Stack Invariant Masking**

Before computing softmax over output logits, the current stack machine state $\\Omega\_t \= (\\text{PC}\_t, \\Sigma\_t, \\Gamma\_t, M\_t)$ is inspected3. Opcodes that violate hard structural invariants are assigned a logit mask of $-\\infty$3. For an opcode $o$ requiring $n\_{\\text{pop}}(o)$ stack arguments, the sampling distribution is constrained via3:

$$\\pi\_\\theta(a\_t \= o \\mid s\_t) \= \\begin{cases} 0 & \\text{if } |\\Sigma\_t| \< n\_{\\text{pop}}(o) \\text{ (Stack Underflow)}, \\\\ 0 & \\text{if } |\\Sigma\_t| \- n\_{\\text{pop}}(o) \+ n\_{\\text{push}}(o) \> S\_{\\text{max}} \\text{ (Stack Overflow)}, \\\\ 0 & \\text{if } \\text{Type}(\\Sigma\_t\[\\text{top}\]) \\neq \\text{ReqType}(o) \\text{ (Type Mismatch)}, \\\\ \\frac{\\exp(z\_o)}{\\sum\_{o' \\in \\mathcal{O}\_{\\text{valid}}} \\exp(z\_{o'})} & \\text{otherwise}. \\end{cases}$$  
This static constraint filtering guarantees that 100% of generated bytecode completions are syntactically valid and trap-free with respect to stack underflows3.

### **Speculative Interleaved Execution and Search Tree Integration**

When the model emits a complete basic block or control statement (such as if, loop, or block), the sandboxed interpreter speculatively executes the block against available test inputs3. If the interpreter triggers a runtime exception or unmapped memory access, the entire basic block branch is pruned, and decoding backtracks to the preceding decision node8.  
For complex algorithmic synthesis, execution guidance is embedded directly into search frameworks such as Tree-of-Thoughts (ToT) and Monte Carlo Tree Search (MCTS)16:  
In the state expansion phase, at tree node $U\_k$ representing state $\\Omega\_k$, the policy samples $K$ candidate bytecode instruction extensions16.  
During execution evaluation, each candidate extension is executed in the VM interpreter16. Nodes that produce identical stack frames $\\Sigma\_t$ and variable states $\\Gamma\_t$ are collapsed into single equivalency classes, drastically reducing search tree width16.  
In rollout value estimation, unfinished branches are evaluated using process reward models or quick unit-test rollouts, steering search toward paths that maximize semantic alignment8.

## **Comprehensive Taxonomy and Comparative Framework**

Achieving efficient, highly accurate synthesis of low-level stack bytecodes requires selecting the appropriate credit assignment paradigm based on task constraints, availability of reference executions, and computational budgets1.

| Credit Assignment Paradigm | Core Optimization Mechanism | Gradient / Reward Signal Granularity | Primary Strengths | Dominant Bottlenecks & Limitations | Exemplary Architectural Implementation |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Outcome-Based Policy Gradient (Standard RLVR)** | Sequence-level return normalization4 | Uniform sequence return $\\hat{A} \\in \[-1, 1\]$ \[cite: 4, 5\] | Minimal implementation complexity; no execution instrumentation required4. | Severe credit smear; high gradient variance; penalizes valid headers4. | GRPO, PPO, REINFORCE++4 |
| **Coverage-Masked Credit Assignment** | Unexecuted token masking based on coverage traces7 | Token-level binary mask $M\_t \\cdot \\hat{A}$ \[cite: 7, 9\] | Prevents dead/unreachable code from receiving spurious credit7. | Cannot disambiguate errors when programs execute to completion4. | StepCoder (FGO)7 |
| **Execution-Grounded Attribution (EGCA)** | Pairwise execution trace alignment against reference solutions4 | Masked token advantage $a\_{i,t}$ concentrated on $T\_{k^\*}$ \[cite: 4\] | Eliminates credit smear completely; isolates exact failure boundary4. | Requires canonical reference traces during RL training4. | EGCA for GRPO4 |
| **Auxiliary Execution Alignment** | Multi-task prediction of variable propagation trajectories1 | Dual-objective $J\_{\\text{gen}} \+ J\_{\\text{align}}$ \[cite: 1\] | Trains policy to internalize runtime variable state transitions1. | Increases training loss complexity and memory footprint1. | CodeRL+1 |
| **Process-Supervised Bi-Level GRPO** | Group-relative step normalizations \+ intra-trajectory shaping8 | Stepwise advantage $\\hat{A}\_{i,g}^{\\text{group}}$ & $\\hat{A}\_{i,g}^{\\text{intra}}$ \[cite: 8\] | Eliminates reward hacking; provides fine-grained dense rewards without PRM drift8. | Requires trace anchor insertion into target code8. | StepCodeReasoner8 |
| **Differentiable Interpretation** | Continuous relaxation of stack pointers and instructions via autograd2 | Exact analytical gradients $\\frac{\\partial \\mathcal{L}}{\\partial \\theta}$ through interpreter2 | Enables direct end-to-end optimization without sampling variance2. | Vanishing gradients; $O(T \\cdot \\Vert{}\\mathcal{O}\\Vert{} \\cdot S^2)$ memory scaling; semantic blurring18. | Differentiable Forth ($\\partial 4$), DMCI, DTM2 |
| **Execution-Guided Search Decoding** | Sandboxed VM execution interleaved with logit masking & MCTS3 | Real-time constraint enforcement at decoding time3 | Guarantees 100% syntactically valid and trap-free bytecode output3. | High inference latency due to VM context switching3. | Interleaved WASM Speculative Decoding3 |

## **Conclusions and Strategic Design Synthesis**

The synthesis of low-level, stack-based bytecode represents a fundamental challenge for autoregressive language models due to the strict coupling between discrete token selections and implicit execution stack states1. Standard outcome-level policy gradient methods fail under these conditions, as sequence-level reward broadcasting induces severe credit smear and penalizes valid algorithmic prefixes4. Overcoming these limitations requires a unified system architecture that integrates execution-grounded credit attribution, process-level shaping advantages, and static constraint masking during inference1.  
For engineering industrial-grade code generation pipelines targeting stack bytecodes such as WebAssembly S-expressions, the evidence indicates a clear hierarchy of implementation practices:  
At the reinforcement learning optimization layer, policy training should combine Execution-Grounded Credit Assignment (EGCA) with Bi-Level GRPO4. By classifying rollouts into priority-ordered failure gates (SYNTAX, CONSTRAINT, LOGIC, CORRECT) and mapping execution trace mismatches to the earliest divergence boundary $k^\*$, the advantage operator zero-masks downstream tokens ($t \> \\max T\_{k^\*}$)4. This protects structural headers and valid variable initializations from negative policy updates4. Concurrently, intermediate execution anchor states must be evaluated using intra-trajectory shaping advantages ($\\hat{A}\_{i,g}^{\\text{intra}}$), ensuring that step-level rewards are conditioned on downstream sequence correctness to prevent reward hacking8.  
At the inference and decoding layer, autoregressive sampling must be tightly coupled with sandboxed virtual machine execution3. Calculating stack arity constraints $n\_{\\text{pop}}(o)$ dynamically and applying static $-\\infty$ logit masks guarantees that the model never emits stack-underflowing or type-mismatched instructions3. Integrating speculative execution within Tree-of-Thoughts or Monte Carlo Tree Search enables early pruning of trap-inducing basic blocks before committing tokens to the KV-cache, drastically improving inference-time search efficiency3.  
Finally, while differentiable interpreters offer theoretical elegance by enabling exact backpropagation through continuous stack relaxations, their $O(T \\cdot |\\mathcal{O}| \\cdot S^2)$ computational overhead and gradient instability over long traces restrict their practical utility2. Soft execution engines should therefore be reserved for training compact, specialized arithmetic or memory-manipulation subroutines2. These primitives can subsequently be integrated as validated macro-instructions within discrete execution-guided autoregressive models, establishing a robust framework for scalable, semantically aligned bytecode synthesis3.

#### **Works cited**

> 1. CODERL+: Improving Code Generation via Reinforce \- arXiv, [https://arxiv.org/pdf/2510.18471](https://arxiv.org/pdf/2510.18471)  
> 2. A Differentiable Meta-Circular Interpreter \- arXiv, [https://arxiv.org/pdf/2606.09930](https://arxiv.org/pdf/2606.09930)  
> 3. Think Anywhere in Code Generation \- arXiv, [https://arxiv.org/html/2603.29957v1](https://arxiv.org/html/2603.29957v1)  
> 4. Execution-Grounded Credit Assignment for GRPO in Code Generation, [https://arxiv.org/pdf/2603.16158](https://arxiv.org/pdf/2603.16158)  
> 5. Outcome-Grounded Advantage Reshaping for Fine-Grained Credit, [https://aclanthology.org/2026.acl-long.1132/](https://aclanthology.org/2026.acl-long.1132/)  
> 6. Tail-Aware Credit Calibration for LLM Reinforcement Learning, [https://www.researchgate.net/publication/408699698\_When\_Implausible\_Tokens\_Get\_Reinforced\_Tail-Aware\_Credit\_Calibration\_for\_LLM\_Reinforcement\_Learning](https://www.researchgate.net/publication/408699698_When_Implausible_Tokens_Get_Reinforced_Tail-Aware_Credit_Calibration_for_LLM_Reinforcement_Learning)  
> 7. StepCoder: Improve Code Generation with Reinforcement Learning, [https://arxiv.org/abs/2402.01391](https://arxiv.org/abs/2402.01391)  
> 8. StepCodeReasoner: Aligning Code Reasoning with Stepwise, [https://arxiv.org/pdf/2605.11922](https://arxiv.org/pdf/2605.11922)  
> 9. StepCoder: Improve Code Generation with Reinforcement Learning, [https://www.semanticscholar.org/paper/StepCoder%3A-Improve-Code-Generation-with-Learning-Dou-Liu/08e84c939b88fc50aaa74ef76e202e61a1ad940b](https://www.semanticscholar.org/paper/StepCoder%3A-Improve-Code-Generation-with-Learning-Dou-Liu/08e84c939b88fc50aaa74ef76e202e61a1ad940b)  
> 10. CodeRL+: Improving Code Generation via Reinforcement with, [https://arxiv.org/html/2510.18471v2](https://arxiv.org/html/2510.18471v2)  
> 11. \[2510.18471\] CodeRL+: Improving Code Generation via ... \- arXiv, [https://arxiv.org/abs/2510.18471](https://arxiv.org/abs/2510.18471)  
> 12. Aligning Code Reasoning with Stepwise Execution Traces via, [https://arxiv.org/abs/2605.11922](https://arxiv.org/abs/2605.11922)  
> 13. Execution-Grounded Credit Assignment for GRPO in Code Generation, [https://www.semanticscholar.org/paper/Execution-Grounded-Credit-Assignment-for-GRPO-in-Kumar-Kumar/dfdacd502be4a341e74b3384d9777d9fdad704f4](https://www.semanticscholar.org/paper/Execution-Grounded-Credit-Assignment-for-GRPO-in-Kumar-Kumar/dfdacd502be4a341e74b3384d9777d9fdad704f4)  
> 14. CodeRL+: Improving Code Generation via Reinforcement with, [https://arxiv.org/html/2510.18471v1](https://arxiv.org/html/2510.18471v1)  
> 15. Process-Supervised Reinforcement Learning for Code Generation, [https://www.researchgate.net/publication/397423372\_Process-Supervised\_Reinforcement\_Learning\_for\_Code\_Generation](https://www.researchgate.net/publication/397423372_Process-Supervised_Reinforcement_Learning_for_Code_Generation)  
> 16. GitHub \- LightChen233/Awesome-Long-Chain-of-Thought-Reasoning, [https://github.com/LightChen233/Awesome-Long-Chain-of-Thought-Reasoning](https://github.com/LightChen233/Awesome-Long-Chain-of-Thought-Reasoning)  
> 17. Daily Papers \- Hugging Face, [https://huggingface.co/papers?q=code%20reasoning](https://huggingface.co/papers?q=code+reasoning)  
> 18. Programming with a differentiable forth interpreter | Request PDF, [https://www.researchgate.net/publication/326682126\_Programming\_with\_a\_differentiable\_forth\_interpreter](https://www.researchgate.net/publication/326682126_Programming_with_a_differentiable_forth_interpreter)  
> 19. Differentiable Tree Operations Promote Compositional Generalization, [https://proceedings.mlr.press/v202/soulos23a/soulos23a.pdf](https://proceedings.mlr.press/v202/soulos23a/soulos23a.pdf)  
> 20. Synthesized Differentiable Programs \- OpenReview, [https://openreview.net/pdf?id=duBR\_dgk\_8M](https://openreview.net/pdf?id=duBR_dgk_8M)  
> 21. Differentiate the Evaluator, Not the Program: An Efficient Runtime, [https://arxiv.org/pdf/2607.03574](https://arxiv.org/pdf/2607.03574)  
> 22. (PDF) Neural Functional Programming \- ResearchGate, [https://www.researchgate.net/publication/309738700\_Neural\_Functional\_Programming](https://www.researchgate.net/publication/309738700_Neural_Functional_Programming)  
> 23. Execution-Grounded Credit Assignment for GRPO in Code Generation, [https://arxiv.org/abs/2603.16158](https://arxiv.org/abs/2603.16158)