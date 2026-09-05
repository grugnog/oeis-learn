# **Continuous-to-Discrete Numeric Constant Binding in Transformer-Based Autoregressive Program Synthesis**

## **1\. Theoretical Foundations of the Continuous-to-Discrete Grounding Gap**

The systematic failure of the OEIS-Learn architecture—wherein the autoregressive WebAssembly Text (WAT) decoder reliably produces the correct computational topology but defaults multiplicative coefficients to unity or uncoupled arbitrary scalars—is an archetype of the continuous-to-discrete literal grounding gap. In neuro-symbolic program synthesis, synthesizing the discrete syntactic skeleton of an algorithm operates under fundamentally different optimization dynamics than binding continuous or wide-domain numerical parameters. The root causes of this collapse reside in the mathematical interaction between cross-attention readout geometry, policy gradient credit assignment under sparse verification, and non-linear encoder feature modulation.

### **Attention Readout Disconnect and the Voronoi Partitioning Bottleneck**

In the baseline system, sequence observations $Y \= \[y\_0, y\_1, \\dots, y\_{19}\]$ are transformed into latent vectors $Z \\in \\mathbb{R}^{20 \\times 256}$. At decoder step $t$, the multi-head cross-attention mechanism computes:

$$A\_{t, j} \= \\text{softmax}\\left(\\frac{Q\_t K\_j^T}{\\sqrt{d\_k}}\\right), \\quad C\_t \= \\sum\_{j=1}^{20} A\_{t, j} V\_j$$  
where $Q\_t \= W\_Q h\_t^{\\text{dec}}$, $K\_j \= W\_K z\_j$, and $V\_j \= W\_V z\_j$. The categorical distribution over the WebAssembly token vocabulary $\\mathcal{V}$ is parameterized via a linear projection head:

$$P(w\_t \\mid w\_{\<t}, Z) \= \\text{softmax}\\left(W\_{\\text{vocab}} h\_t^{\\text{out}} \+ b\_{\\text{vocab}}\\right)$$  
When the dynamic Abstract Syntax Tree (AST) grammar mask restricts $w\_t$ to an immediate numerical constant (such as the integer operand following an i64.const instruction), the decoder is forced to project the continuous context vector $C\_t$ into an exact discrete class index.

Linear multi-head dot-product attention computes a soft convex combination of encoder representations. While convex combinations can smoothly route semantic activations across sequence positions, they cannot compute non-linear arithmetic invariants—such as the ratio of finite differences $\\frac{\\Delta y\_i}{\\Delta n}$ or the determinant of a recurrence system—directly through bilinear query-key inner products. Because the continuous representations of numeric values lie on curved geometric manifolds within the latent space $Z$, a linear classification head $W\_{\\text{vocab}} \\in \\mathbb{R}^{\\vert{}\\mathcal{V}\\vert{} \\times d}$ must partition the decoder hidden space into high-dimensional Voronoi polyhedra.

Under any variance or uncertainty in $C\_t$, cross-entropy loss severely penalizes predictions that place probability mass outside the exact target label, while assigning uniform penalty to all incorrect discrete tokens regardless of numerical proximity. The network minimizes expected cross-entropy risk by collapsing the output distribution to the empirical mode of the constant tokens observed across the training distribution. In procedural code and integer sequence corpora, identity constants such as 0 and 1 appear with overwhelming frequency as loop bounds, stack initializers, and multiplicative base cases, making them the entropy-minimizing centroids of the categorical head.

### **Credit Assignment Pathology under Group Relative Policy Optimization**

The reinforcement learning objective exacerbates this representational disconnect. Group Relative Policy Optimization (GRPO) samples a group of $K$ candidate programs $\\{P\_1, P\_2, \\dots, P\_K\\}$ from the prior policy $\\pi\_{\\theta\_{\\text{old}}}$ for a given sequence $Y$. The advantage $\\hat{A}\_k$ for each rollout is computed using normalized group outcomes:

$$\\hat{A}\_k \= \\frac{R\_k \- \\text{mean}(\\{R\_j\\}\_{j=1}^K)}{\\text{std}(\\{R\_j\\}\_{j=1}^K) \+ \\epsilon}$$  
Under an exact execution-based reward function:

$$R(P, Y) \= \\begin{cases} 1 & \\text{if } \\forall n \\in \\{0, \\dots, 19\\}, \\, P(n) \= y\_n \\\\ 0 & \\text{otherwise} \\end{cases}$$  
the likelihood of sampling the exact tuple of discrete constants required to satisfy all twenty terms decreases exponentially with the dynamic range of the target coefficients. For an affine sequence governed by $a(n) \= 5n \+ 2$, the decoder may generate an AST with the correct topological structure:

$$\\text{local.get } \\$n \\to \\text{i64.extend\\\_i32\\\_u} \\to \\text{i64.const } C\_1 \\to \\text{i64.mul} \\to \\text{i64.const } C\_2 \\to \\text{i64.add}$$  
If the policy samples $C\_1 \\in \\{1, 2, 3, 7\\}$ and $C\_2 \\in \\{0, 1, 2\\}$, every rollout in the group fails the strict equivalence test, resulting in $R\_1 \= R\_2 \= \\dots \= R\_K \= 0$. Consequently, the group standard deviation collapses ($\\text{std}(\\{R\\}) \= 0$), yielding identical zero advantages ($\\hat{A}\_k \= 0$) across the batch.

With vanishing policy gradients on the numeric token positions, optimization is dominated by the auxiliary Supervised Fine-Tuning (SFT) loss. Because the SFT loss trains on static corpus distributions where standard AST templates predominantly utilize identity transitions, the policy gradients fail to overcome the SFT inductive bias. The system falls into an AST idiom trap: structural operators generalize due to rich grammatical feedback, while constant slots collapse to default literals.

### **Modulatory Bottlenecks in Hierarchical FiLM Fusion**

The continuous encoder uses a two-stage Hierarchical Feature-wise Linear Modulation (FiLM) framework to combine signed log-magnitude features $S\_1$, Fourier residue phase spectrums $S\_2$, and finite differences with $p$-adic valuations $S\_3$. FiLM modifies intermediate representations via learned affine transformations:

$$\\mathbf{h}\_{\\text{fused}} \= \\gamma(S\_{\\text{condition}}) \\odot S\_{\\text{target}} \+ \\beta(S\_{\\text{condition}})$$  
Although FiLM is effective for visual reasoning and style conditioning, applying multiplicative modulation directly across continuous arithmetic signals introduces severe non-linear distortion. When $S\_2$ (the complex exponential phase spectrum $\\exp(2\\pi i (y \\bmod m)/m)$) modulates the log-scale representations $S\_1$, it creates an oscillating, non-convex latent surface where distance in latent space no longer corresponds monotonically to differences in sequence growth rates.

Because cross-attention queries must extract scalar multipliers through linear inner products, this non-linear entanglement prevents linear attention projections from reading sequence derivatives directly from the encoder representations.

## **2\. Decoupled Neuro-Symbolic Architectures: Program Topology vs. Numeric Solvers**

To bypass the continuous-to-discrete bottleneck, modern symbolic regression and neuro-symbolic program synthesis systems increasingly decouple discrete computational topology generation from continuous parameter estimation. Instead of forcing a language model to guess numerical constants autoregressively, the generative model outputs an abstract program skeleton containing constant placeholders (const\_?), leaving coefficient binding to specialized deterministic or continuous numerical solvers.

### **Paradigms in Contemporary Symbolic Regression**

The division between end-to-end tokenization and decoupled two-stage solving has been explored extensively across several major architectures:

Deep Symbolic Regression (DSR) and its genetic-algorithmic extension DSO utilize an autoregressive recurrent neural network trained via risk-seeking policy gradients to emit the pre-order traversal of mathematical expressions. All numerical leaves in the parse tree are emitted as generic constant placeholders. Once a skeleton is complete, a non-linear continuous optimizer—specifically the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm—is executed to fit the constants against the target numerical values.

Neural Symbolic Regression that Scales (NeSymReS) adapts this principle to large Transformer architectures. NeSymReS pre-trains an encoder-decoder network on synthetic functional trees, tasks the decoder with predicting expression skeletons, and executes multi-start BFGS post-hoc to recover continuous parameters.

Conversely, End-to-End Symbolic Regression (E2E-SR) and SymFormer challenge pure post-hoc optimization. E2E-SR demonstrates that while BFGS is effective on low-dimensional smooth landscapes, it frequently converges to poor local minima on deeper, non-linear ASTs when initialized from random values. E2E-SR trains Transformers to emit expressions and constants simultaneously, using predicted constants as warm-start initializations for post-hoc BFGS refinement.

SymFormer formalizes this synergy through a dual-head architecture: the symbolic head predicts expression tokens while a continuous regression head simultaneously predicts continuous constants, ensuring full end-to-end differentiability during backpropagation.

### **Solver Mechanics for WebAssembly Execution**

In the OEIS-Learn setting, applying a decoupled framework requires mapping emitted WebAssembly skeletons containing $k$ free integer placeholders $\\mathbf{C} \= \[c\_1, c\_2, \\dots, c\_k\] \\in \\mathbb{Z}^k$ to target execution traces $Y \= \[y\_0, \\dots, y\_{19}\]$. Depending on the algebraic structure of the generated AST, three distinct solver paradigms can be deployed.

```
+-----------------------------------------------------------------------------+
|                         Decoupled Solver Architecture                       |
+-----------------------------------------------------------------------------+
                                       |
                                       v
               +-----------------------------------------------+
               | Tri-Stream Encoder & Grammar Decoder Rollout  |
               | (Emits WAT Program Skeleton with 'const_?')   |
               +-----------------------------------------------+
                                       |
                                       v
               +-----------------------------------------------+
               |         AST Linearity & Operator Parser       |
               +-----------------------------------------------+
                                       |
         +-----------------------------+-----------------------------+
         |                                                           |
         v                                                           v
  [Linear/Affine Trace]                                    [Non-Linear/Control Flow]
         |                                                           |
         v                                                           v
+-------------------------------+                         +---------------------+
| Exact Diophantine Solver      |                         | Satisfiability      |
| (Hermite Normal Form / ILP)   |                         | Modulo Theories     |
| Solving: A * C = Y over Z     |                         | (Z3 QF_BV / QF_NIA) |
+-------------------------------+                         +---------------------+
         |                                                           |
         +-----------------------------+-----------------------------+
                                       |
                                       v
               +-----------------------------------------------+
               |    Grounded, Executable WebAssembly Module    |
               +-----------------------------------------------+
```

When placeholders appear linearly within the execution trace, such that the program output is an affine combination of computed terms:

$$P\_{\\mathbf{C}}(n) \= c\_1 f\_1(n) \+ c\_2 f\_2(n) \+ \\dots \+ c\_k f\_k(n) \+ c\_0$$  
the constant binding task simplifies to a system of linear Diophantine equations. By executing the partial program skeleton for each input index $n \\in \\{0, \\dots, 19\\}$ with basis unit vectors, the system constructs the coefficient matrix $A \\in \\mathbb{Z}^{20 \\times k}$ where $A\_{n, j} \= f\_j(n)$. The exact integer solution vector $\\mathbf{C} \\in \\mathbb{Z}^k$ satisfying:

$$\\mathbf{A} \\mathbf{C} \= \\mathbf{Y}$$  
can be determined in polynomial time via Hermite Normal Form (HNF) decomposition or standard Integer Linear Programming (ILP) solvers. The Diofantos engine established that using exact Diophantine solvers to resolve constants in recurrence relations dramatically outperforms continuous numerical regression over integer sequence benchmarks such as the OEIS.

When placeholders appear within non-linear operations, integer modular arithmetic (i64.rem\_u), bitwise shifts (i64.shl, i64.shr\_u), or conditional branches (if ... else), gradient-based optimization fails entirely due to vanishing derivatives and step discontinuities. Under these conditions, the skeleton is formulated as a Satisfiability Modulo Theories (SMT) problem under CounterExample-Guided Inductive Synthesis modulo Theories, or CEGIS(T). The parameterized WAT program is lowered to an SMT-LIB2 formula over the quantifier-free theory of Fixed-Size BitVectors (QF\_BV) or Non-Linear Integer Arithmetic (QF\_NIA):

$$\\exists \\mathbf{C} \\in (\\mathbb{Z}/2^{64}\\mathbb{Z})^k \\quad \\text{such that} \\quad \\bigwedge\_{n=0}^{19} \\left( \\llbracket P\_{\\mathbf{C}} \\rrbracket(n) \= y\_n \\right)$$  
Modern SMT solvers such as Z3 and CVC4 resolve these finite bit-vector constraints across small parameter spaces ($k \\le 4$) within tens of milliseconds, completely insulating the transformer policy from the burden of arithmetic constant discovery.

For smooth, non-branching ASTs composed exclusively of arithmetic operators (i64.add, i64.mul, i64.sub), continuous relaxation offers an alternative route. The integer stack semantics of the WebAssembly bytecode are temporarily translated into double-precision floating-point instructions (f64), enabling standard Levenberg-Marquardt or BFGS routines to optimize the continuous least-squares error:

$$\\mathcal{L}(\\mathbf{C}) \= \\sum\_{n=0}^{19} \\left( P\_{\\mathbf{C}}^{\\mathbb{R}}(n) \- y\_n \\right)^2$$  
Upon convergence to a local minimum $\\mathbf{c}^\* \\in \\mathbb{R}^k$, the parameters are mapped back to discrete space via rounding $\\widehat{\\mathbf{C}} \= \\lfloor \\mathbf{c}^\* \\rceil$ and verified through discrete WebAssembly execution.

| Architectural Paradigm | Inference Latency | Target Domain | AST Expressiveness | Completeness Guarantees | Policy Gradient (GRPO) Fit |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Pure End-to-End Autoregressive** | Minimal ($O(T)$ forward steps) | Discrete Token Vocabulary | Unconstrained syntax | No guarantees; empirical mode collapse | Severe reward sparsity and zero advantage |
| **Two-Stage: SMT / CEGIS(T) (Z3)** | Variable (5 ms to 2 s timeout) | Bit-exact integers (i32, i64) | Full WebAssembly (shifts, bitwise, branches) | Complete: deterministically returns SAT or UNSAT | High: provides valid reward signals for correct skeletons |
| **Two-Stage: Diophantine / ILP** | Very Low (\<10 ms via HNF) | Exact Ring of Integers $\\mathbb{Z}$ \[cite: 19\] | Linear / affine parameter occurrences only | Globally exact if system is consistent | High: deterministic positive reinforcement |
| **Two-Stage: Continuous BFGS** | Moderate (20 ms to 150 ms) | Continuous $\\mathbb{R}$ relaxed to $\\mathbb{Z}$ \[cite: 1, 3\] | Differentiable arithmetic only (no bitwise/modulo) | Incomplete: vulnerable to non-convex local minima | Moderate: requires proxy continuous loss |

## **3\. Pointer Networks and Continuous-Valued Decoder Heads**

While two-stage decoupled architectures offload numeric binding to downstream solvers, they discard end-to-end gradient signals from constant values back to the encoder. Retaining end-to-end differentiability while mitigating categorical cross-entropy pathology requires rethinking how the autoregressive decoder outputs numerical values.

### **Dual Symbolic-Numeric Decoder Heads**

Following the principles established in SymFormer, the decoder’s classification vocabulary $\\mathcal{V}$ is restricted purely to syntactic grammar tokens, terminal identifiers, and a single abstract numerical constant marker, const\_scalar. The final transformer hidden state $h\_t^{\\text{dec}} \\in \\mathbb{R}^d$ is routed into two parallel output heads:

```
                  +----------------------------------------------+
                  |         Decoder Hidden State h_t_dec         |
                  +----------------------------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
     +-------------------------------+       +-------------------------------+
     |     Grammar-Constrained       |       |     Continuous Parameter      |
     |       Syntax Head             |       |        Regression Head        |
     +-------------------------------+       +-------------------------------+
                     |                                       |
                     v                                       v
      Softmax Logits over AST Tokens          Gaussian Parameters: mu_t, sigma_t
    (Restricted to 'const_scalar' if                 (Optimized via NLL Loss;
       AST grammar requires literal)             Evaluated via Integer Rounding)
```

The symbolic head computes categorical logits over AST syntax:

$$p\_{\\text{sym}}(w\_t) \= \\text{softmax}\\left(W\_{\\text{sym}} h\_t^{\\text{dec}} \+ b\_{\\text{sym}}\\right)$$  
while the numerical regression head parameterizes a continuous Gaussian density over the real line:

$$\[\\mu\_t, \\log \\sigma\_t^2\] \= \\text{MLP}\_{\\text{num}}\\left(h\_t^{\\text{dec}}\\right)$$  
During autoregressive generation, whenever the grammar-guided mask dictates that the next token must be an immediate constant for an instruction like i64.const, the categorical distribution is clamped to const\_scalar with probability 1\. The numerical head then outputs $\\mu\_t$, which is projected to an integer literal via deterministic rounding:

$$\\widehat{C}\_t \= \\lfloor \\mu\_t \\rceil$$  
During training, the continuous head is supervised via Gaussian Negative Log-Likelihood (NLL) against ground-truth program constants $C\_t^\*$:

$$\\mathcal{L}\_{\\text{num}} \= \\frac{(C\_t^\* \- \\mu\_t)^2}{2\\sigma\_t^2} \+ \\frac{1}{2}\\log \\sigma\_t^2$$  
This formulation preserves gradient flow. Because the continuous loss penalizes predictions proportionally to their metric distance $\\vert{}C\_t^\* \- \\mu\_t\\vert{}^2$, gradients propagate smoothly through the decoder’s cross-attention layers into the continuous encoder, rewarding outputs that track the proper order of magnitude even before the exact integer value is locked in.

### **Continuous Number Encoding via xVal**

The xVal paradigm replaces discrete number tokenization by encoding numeric scalars as continuous vectors. Rather than assigning dedicated vocabulary embeddings to different integers, a single designated embedding token $\\mathbf{e}\_{\\text{num}} \\in \\mathbb{R}^d$ is scaled directly by the numerical value $x$:

$$\\mathbf{E}(x) \= x \\cdot \\mathbf{e}\_{\\text{num}}$$  
In the synthesis decoder, numeric literal prediction is handled by an unnormalized linear readout head:

$$\\widehat{y}\_t \= \\mathbf{w}\_{\\text{val}}^T h\_t^{\\text{dec}} \+ b\_{\\text{val}}$$  
This transformation makes the transformer continuous with respect to numeric constant emission. Unlike categorical cross-entropy, which treats predicting $4$ instead of $5$ with the same cross-entropy loss as predicting $999$, continuous number decoders preserve metric topology. The model receives informative error gradients throughout training, preventing the flat optimization plateaus that cause categorical heads to default to identity constants.

### **Multi-Token Positional Number Encodings**

When system constraints necessitate purely discrete token vocabularies, monolithic integer tokenization must be replaced with structured multi-token decompositions to eliminate out-of-vocabulary failures and model arithmetic carry mechanics. Two structural encodings are prominent:

In scientific mantissa-exponent tokenization, every integer constant $C$ is decomposed into a three-token sequence consisting of sign, a normalized four-digit mantissa, and a power-of-ten exponent:

$$C \= s \\times m \\times 10^e, \\quad s \\in \\{+, \-\\}, \\, m \\in \\{0, \\dots, 9999\\}, \\, e \\in \\{0, \\dots, 18\\}$$  
The dynamic grammar forces the decoder, upon emitting i64.const, to sequentially generate these three tokens. By decomposing scale from magnitude, cross-attention can bind the exponent token directly from log-magnitude sequence features $S\_1$ before determining the finer mantissa digits.

In positional base-10 digit-wise tokenization, integers are emitted as sequential digits from most significant to least significant, terminated by a delimiter:

$$\\text{i64.const } \\to \[s\] \\to d\_k \\to d\_{k-1} \\to \\dots \\to d\_0 \\to \\langle \\text{end\\\_num} \\rangle$$  
By unrolling numbers into digit streams, the autoregressive self-attention layers can condition the prediction of lower-order digits on higher-order scale choices, mimicking standard multi-digit arithmetic operations and preventing out-of-vocabulary representation collapse.

### **Pointer-Generator Attention for Direct Literal Copying**

In integer sequence analysis, required program constants often correspond directly to values present in the input sequence (e.g., initial conditions $y\_0$, offsets $y\_1 \- y\_0$, or common divisors). A standard generative decoder must reconstruct these values from continuous embeddings through its projection layers.

A pointer-generator architecture resolves this redundancy by allowing the model to choose between generating a constant from its vocabulary or copying an integer directly from the input sequence.

```
                +-------------------------------------------------------------+
                |                Decoder State h_t_dec & Context C_t          |
                +-------------------------------------------------------------+
                                               |
                                               v
                               +-------------------------------+
                               | Generation Probability Router |
                               |   p_gen = sigmoid(W * [h, C]) |
                               +-------------------------------+
                                               |
                       +-----------------------+-----------------------+
                       |                                               |
                       v                                               v
         +---------------------------+                   +---------------------------+
         | Multiplied by p_gen       |                   | Multiplied by (1 - p_gen) |
         +---------------------------+                   +---------------------------+
                       |                                               |
                       v                                               v
         +---------------------------+                   +---------------------------+
         |     Vocabulary Head       |                   |    Pointer Head           |
         | P_vocab = Softmax(W * h)  |                   | Softmax Cross-Attention   |
         | (Emits AST Grammar Syntax)|                   | over Input Literals Y     |
         +---------------------------+                   +---------------------------+
                       |                                               |
                       +-----------------------+-----------------------+
                                               |
                                               v
                               +-------------------------------+
                               |  Unified Probability Mass:    |
                               |  P(w) = p_gen * P_vocab(w)    |
                               |    + (1 - p_gen) * Sum A_t,j  |
                               +-------------------------------+
```

The generation probability $p\_{\\text{gen}} \\in \[0, 1\]$ is dynamically estimated from the decoder state and context vector:

$$p\_{\\text{gen}} \= \\sigma\\left(W\_h h\_t^{\\text{dec}} \+ W\_c C\_t \+ b\_p\\right)$$  
The final probability assigned to any literal token $w$ combines vocabulary logits and cross-attention weights over the input sequence:

$$P(w) \= p\_{\\text{gen}} P\_{\\text{vocab}}(w) \+ (1 \- p\_{\\text{gen}}) \\sum\_{j: y\_j \= w} A\_{t, j}$$  
When generating structural AST syntax, the grammar mask forces $p\_{\\text{gen}} \\to 1$. When generating constants, the pointer head allows the model to copy observed sequence terms directly into instruction operands, completely bypassing the continuous-to-discrete classification bottleneck.

| Numerical Output Formulation | Vocabulary Footprint | Differentiability | Arithmetic Metric Sensitivity | Out-of-Distribution Scaling |
| :---- | :---- | :---- | :---- | :---- |
| **Monolithic Discrete Tokens** | Large ($\\vert{}\\mathcal{V}\\vert{} \> 10^4$) | Non-differentiable (Categorical CE) | Zero (Hamming/0-1 step loss) | Fails completely on unseen constants |
| **Digit-by-Digit Positional** | Compact (10 digit tokens \+ signs) | Sequential discrete steps | High across digit positions | High (generalizes across arbitrary lengths) |
| **Mantissa-Exponent Triplet** | Moderate (\~10,000 mantissas, 200 exps) | Sequential discrete steps | Decouples scale from relative magnitude | High across vast dynamic ranges ($10^{\\pm 100}$) |
| **SymFormer Dual Head (Gaussian)** | Single token (const\_scalar) | Fully Differentiable (NLL Loss) | Continuous $L\_2$ metric preserving | Bounded by regression head output capacity |
| **xVal Continuous Projection** | Zero vocabulary expansion | Fully Differentiable (MSE Loss) | Linear metric preserving | High; scales continuously via scalar multipliers |
| **Pointer-Generator Attention** | Dynamic input length mapping | Differentiable attention pooling | Exact literal identity matching | Perfect for in-context sequence values |

## **4\. Representational Inductive Biases in the Continuous Neural Encoder**

The failure of cross-attention to resolve numerical constants points to representational bottlenecks in the Tri-Stream Encoder. To make integer scale, polynomial degree, and recurrence slopes linearly accessible to the decoder's cross-attention mechanisms, the encoder's geometric representations must be realigned.

### **Linearizing Sequence Slopes via Explicit Difference Ratios**

The Newton forward-difference formulation establishes that any sequence generated by a polynomial $P(n)$ of degree $d$ can be uniquely expressed in terms of its forward differences evaluated at the origin:

$$P(n) \= \\sum\_{k=0}^d \\binom{n}{k} \\Delta^k y\_0 \= y\_0 \+ n \\Delta y\_0 \+ \\frac{n(n-1)}{2\!} \\Delta^2 y\_0 \+ \\dots$$  
In the baseline architecture, forward differences $\\Delta y\_i$ and $\\Delta^2 y\_i$ are concatenated with $p$-adic valuations inside stream $S\_3$ and modulated through non-linear FiLM layers. This forces the cross-attention queries to untangle linear differences from modular structures.

Linear readability can be restored by computing an explicit, unmodulated difference quotient tensor across adjacent indices:

$$D\_i^{(1)} \= y\_{i+1} \- y\_i, \\quad D\_i^{(2)} \= \\frac{y\_{i+2} \- 2y\_{i+1} \+ y\_i}{2}, \\quad \\rho\_i \= \\log \\left\\vert{} \\frac{y\_{i+1} \+ \\text{sign}(y\_{i+1})\\epsilon}{y\_i \+ \\text{sign}(y\_i)\\epsilon} \\right\\vert{}$$  
Feeding these raw differences directly through linear projection layers into the cross-attention key-value matrices ensures that for any linear sequence $a(n) \= m \\cdot n \+ b$, the slope $m \= D\_i^{(1)}$ is represented as a static constant across all positions $i \\in \\{0, \\dots, 18\\}$. The cross-attention query for an i64.mul constant can then isolate $m$ through a single dot-product operation.

### **Harmonic Encodings: Prime Fourier Embeddings vs. FoNE**

Recent analyses of arithmetic representations demonstrate that language models naturally develop Fourier-like periodic features when learning numbers, but standard base-10 or unconstrained moduli introduce severe spectral interference.

Fourier Number Embeddings (FoNE) project real values into a multi-frequency basis using base-10 scaling:

$$\\text{FoNE}(x) \= \\left\[ \\cos\\left(\\frac{2\\pi x}{10^k}\\right), \\sin\\left(\\frac{2\\pi x}{10^k}\\right) \\right\]\_{k=1}^K$$  
While FoNE avoids token fragmentation and accelerates training on multi-digit decimal addition, base-10 periodicities entangle composite prime factors (namely 2 and 5), forcing multi-layer decoders to disentangle these signals internally.

Prime Fourier Embeddings (PFE) resolve this entanglement by mapping numbers onto an orthogonal prime basis derived from the harmonic analysis of rational numbers:

$$\\text{PFE}(x) \= \\bigoplus\_{p \\in \\mathcal{P}} \\bigoplus\_{d=1}^D \\left\[ \\cos\\left(\\frac{2\\pi x}{p^d}\\right), \\sin\\left(\\frac{2\\pi x}{p^d}\\right) \\right\]$$  
Because distinct prime fields are mathematically orthogonal, any linear projection matrix acting on PFE representations decomposes into a block-diagonal operator:

$$\\text{PFE}(x) \= \\bigoplus\_{p \\in \\mathcal{P}} \\bigoplus\_{d=1}^D \\left\[ \\cos\\left(\\frac{2\\pi x}{p^d}\\right), \\sin\\left(\\frac{2\\pi x}{p^d}\\right) \\right\]$$  
This structure isolates modular residues into independent channels, preventing cross-channel interference and allowing linear cross-attention heads to read out divisibility, step periodicity, and modular factors directly.

```
+-----------------------------------------------------------------------------+
|                      Augmented Tri-Stream Architecture                      |
+-----------------------------------------------------------------------------+
                                       |
     +---------------------------------+---------------------------------+
     |                                 |                                 |
     v                                 v                                 v
+------------------------+  +------------------------+  +------------------------+
| Stream 1: Scale        |  | Stream 2: Harmonics    |  | Stream 3: Invariants   |
| Signed Log-Magnitude   |  | Prime Fourier          |  | Newton Forward         |
| & Continuous xVal      |  | Embeddings (PFE) over  |  | Differences & Rational |
| Scalar Projections     |  | Orthogonal Primes      |  | Step-Quotients         |
+------------------------+  +------------------------+  +------------------------+
     |                                 |                                 |
     +---------------------------------+---------------------------------+
                                       |
                                       v
               +-----------------------------------------------+
               | Direct Concatenation & Multi-Head Self-Attn   |
               | (Preserves Linear Separability; Replaces FiLM)|
               +-----------------------------------------------+
                                       |
                                       v
               +-----------------------------------------------+
               | Prepended Global Latent Summary Tokens:       |
               | [z_affine, z_poly, z_geom, z_0, ..., z_19]    |
               +-----------------------------------------------+
```

### **Prepending Global Latent Summary Tokens**

The baseline encoder outputs a sequence of position-specific embeddings $Z \\in \\mathbb{R}^{20 \\times 256}$, requiring the decoder's cross-attention mechanism to integrate information across all 20 tokens to compute sequence-wide properties.

This global pooling bottleneck can be eliminated by prepending specialized summary tokens to the encoder input sequence, which aggregate global sequence invariants through bidirectional self-attention:

1. The affine summary token $\\mathbf{z}\_{\\text{affine}}$ is supervised during pre-training using auxiliary regression objectives to output sequence-wide linear regression parameters:

$$\\widehat{m} \= \\frac{\\sum\_{i=0}^{19} (i \- \\bar{i})(y\_i \- \\bar{y})}{\\sum\_{i=0}^{19} (i \- \\bar{i})^2}, \\quad \\widehat{b} \= \\bar{y} \- \\widehat{m} \\bar{i}$$

1. The polynomial summary token $\\mathbf{z}\_{\\text{poly}}$ pools higher-order finite differences $\\Delta^2 Y, \\Delta^3 Y$ to encode polynomial curvature.  
2. The geometric summary token $\\mathbf{z}\_{\\text{geom}}$ tracks sequence-level quotients $\\log |y\_{i+1}/y\_i|$ to encode exponential bases and recurrence factors.

When the decoder's grammar mask indicates a constant slot, cross-attention can route directly to these summary tokens, providing direct access to global sequence properties without requiring multi-token pooling.

## **5\. Curriculum Learning, Synthetic Data Augmentation, and Policy Regularization**

When models are trained predominantly on standard mathematical libraries or canonical OEIS entries, their optimization paths are distorted by distribution bias. Because human-authored programs favor unit steps, small counters, and normalized increments, training on unperturbed programs encourages policies to memorize these default idioms.

### **Combating Idiom Memorization via Randomized Affine Perturbations**

To force the decoder to condition its constant predictions on encoder features rather than language priors, training datasets must break the correlation between program structure and specific constant values. This is achieved by generating randomized AST skeletons and sweeping their literal constants across broad numerical ranges:

1. An algorithmic AST skeleton $P\_{\\text{skel}}(n)$ is sampled from the grammar (e.g., polynomial expansions, nested loops, conditional parity branches).  
2. For every constant node $c\_k$ within the AST, replacement scalars are drawn from log-uniform distributions spanning multiple orders of magnitude:

$$c\_{\\text{scale}} \\sim \\pm 10^{\\mathcal{U}(0, 6)}, \\quad c\_{\\text{offset}} \\sim \\mathcal{U}(-10^5, 10^5)$$

1. The entire computational graph is subjected to random affine scaling:

$$c\_{\\text{scale}} \\sim \\pm 10^{\\mathcal{U}(0, 6)}, \\quad c\_{\\text{offset}} \\sim \\mathcal{U}(-10^5, 10^5)$$  
This transformation exposes the network to identical AST topologies paired with thousands of different coefficient combinations. If the model predicts an identity constant such as $C\_{\\text{mul}} \= 1$, the resulting loss is catastrophic. To minimize cross-entropy and regression error, the model is forced to route cross-attention directly to the encoder's slope and difference features.

### **Dense Surrogate Execution Rewards for GRPO**

Binary 0/1 execution rewards create flat optimization landscapes where group variance vanishes whenever all sampled rollouts fail. To provide continuous learning signals, the binary reward is replaced with a **Continuous Log-Distance Reward**:

$$R\_{\\text{dense}}(P, Y) \= \\frac{1}{20} \\sum\_{n=0}^{19} \\frac{1}{1 \+ \\log\_{10}(|P(n) \- y\_n| \+ 1)}$$  
This reward metric structures the optimization landscape into smooth, informative stages:

* Incorrect AST skeletons (such as generating exponential growth for a linear sequence) produce rapid divergence, driving $R\_{\\text{dense}} \\to 0$.  
* Skeletons with the correct topology ($P(n) \= C\_1 n \+ 2$) but imprecise slopes ($C\_1 \= 4$ instead of $5$) receive substantial partial credit ($R\_{\\text{dense}} \\approx 0.70$ compared to $R\_{\\text{dense}} \\approx 0.15$ for a flatline prediction).

In group relative updates, rollouts that approximate the true slope achieve higher rewards than those collapsing to unity. This guarantees non-zero reward variance ($\\text{std}(\\{R\\}) \> 0$) across the sampling group, providing consistent gradient updates that pull constant predictions toward their ground-truth values.

### **Progressive Self-Learning Curriculum (PSL / DASRIS)**

Following the Progressive Self-Learning (PSL) framework for integer sequence discovery, synthesis models should be trained through a multi-stage curriculum that gradually increases structural complexity.

| Curriculum Stage | Program Grammar Restrictions | Constant Sampling Domain | Training Objective | Core Representational Target |
| :---- | :---- | :---- | :---- | :---- |
| **Stage 1: Linear & Polynomial Grounding** | Loop-free affine and pure polynomial WAT instructions: i64.mul, i64.add, i64.const \[cite: 3, 5\] | Wide-range integers: $C \\in \[-10^5, 10^5\]$ sampled log-uniformly | SFT with Continuous Regression Loss ($L\_2$ / NLL) | Aligning cross-attention queries with finite-difference streams $\\Delta^k Y$ \[cite: 6, 27\] |
| **Stage 2: Linear Recurrences** | Fixed-order recurrences: $u\_n \= \\sum\_{j=1}^d c\_j u\_{n-j}$ with basic stack caching | Small-to-medium coefficients: $c\_j \\in \[-50, 50\]$ \[cite: 39, 40\] | SFT co-training \+ Dense Reward GRPO ($R\_{\\text{dense}}$) | Conditioning recurrence coefficients on $p$-adic and Prime Fourier features |
| **Stage 3: Full Turing-Complete WAT** | Conditionals (if/else), blocks, loops (br\_if), bitwise operators, modular arithmetic | Unconstrained integer literals and bitmasks | GRPO with SMT/ILP Fallback \+ CGI | Decoupling complex control-flow search from constant verification |
| **Stage 4: Domain Adaptation on OEIS** | Unconstrained program space on real OEIS sequences | Real OEIS sequence constants and recurrence signatures | Progressive Self-Learning: discovered programs enter training buffer | Generalizing from synthetic functional priors to human mathematical formulations |

## **6\. Implementation Roadmap and Recommendations for OEIS-Learn**

To systematically resolve the continuous-to-discrete constant grounding gap in OEIS-Learn, modifications should be deployed across three sequential engineering phases.

```
+-----------------------------------------------------------------------------+
|                      Phased Implementation Architecture                     |
+-----------------------------------------------------------------------------+
                                       |
     +---------------------------------+---------------------------------+
     |                                 |                                 |
     v                                 v                                 v
+-------------------------+ +-------------------------+ +---------------------+
| Phase 1: Short-Term     | | Phase 2: Mid-Term       | | Phase 3: Long-Term  |
| Decoupled Solver        | | Encoder/Decoder         | | Pretraining &       |
| Integration             | | Architectural Upgrade   | | Objective Alignment |
|                         | |                         | |                     |
| * 'const_?' AST Grammar | | * Dual SymFormer Head   | | * Dense Log-Distance|
|   Mask Token            | |   (Continuous Gaussian) | |   GRPO Rewards      |
| * Linearity Classifier  | | * Newton Difference     | | * Affine Data Sweeps|
| * Exact HNF Diophantine | |   Quotient Streams      | |   (Dynamic Scaling) |
|   & Z3 SMT Fallback     | | * Global Summary Tokens | | * Progressive OEIS  |
| * Solved Constants Roll-| |   (z_affine, z_geom)    | |   Self-Learning     |
|   back into GRPO Buffer | | * PFE Prime Harmonics   | |   Curriculum        |
+-------------------------+ +-------------------------+ +---------------------+
```

### **Phase 1: Short-Term Decoupled Solver Integration**

The primary immediate objective is to prevent policy gradient starvation under GRPO without requiring full architectural retraining.

1. **Placeholder Grammar Extension**: Modify the decoder's dynamic AST grammar mask to permit emitting an untyped constant placeholder token i64.const\_?.  
2. **AST Linearity Classification and Dispatch**: Upon sampling a complete program skeleton containing $k$ placeholders ($k \\le 4$), parse the AST to inspect placeholder positions:  
   * If all placeholders appear linearly in the execution trace, formulate the linear Diophantine system $\\mathbf{A} \\mathbf{C} \= \\mathbf{Y}$ and solve for $\\mathbf{C} \\in \\mathbb{Z}^k$ via Hermite Normal Form (HNF) decomposition using a native C++ runtime extension.  
   * If placeholders appear within modular expressions, shifts, or conditionals, lower the program to an SMT problem and invoke Z3 using the QF\_BV logic with a 250-millisecond execution timeout.  
3. **GRPO Advantage Replacement**: If the solver finds a valid integer solution $\\mathbf{C}^\*$, splice the concrete values back into the rollout program and set the reward $R\_k \= 1.0$. This provides positive reinforcement to rollouts that identify the correct computational skeleton, allowing structural exploration to proceed unhindered by literal guessing.

### **Phase 2: Mid-Term Encoder and Decoder Architectural Refactoring**

The secondary objective is to upgrade the neural backbone to support end-to-end continuous numeric grounding.

1. **Dual Symbolic-Numeric Decoder Head**:  
   * Prune all multi-digit numeric tokens from the decoder vocabulary, retaining only the single abstract token const\_scalar.  
   * Attach a continuous regression head (a three-layer MLP with GELU activations) to the decoder’s final hidden state, parameterizing a Gaussian distribution $\[\\mu\_t, \\log \\sigma\_t^2\]$.  
   * Supervise the continuous head with Gaussian Negative Log-Likelihood against ground-truth program constants, scaling the loss by an auxiliary weight $\\lambda \= 0.5$.  
   * During inference, clamp the numeric prediction to $\\lfloor \\mu\_t \\rceil$, serialize the integer into standard WebAssembly LEB128 binary encoding, and execute.  
2. **Representational Upgrades in the Encoder**:  
   * Retire the two-stage Hierarchical FiLM module in favor of direct vector concatenation followed by full multi-head self-attention, preventing non-linear modulatory distortion of scale features.  
   * Augment the input with an explicit **Newton Difference Stream** computing $D^{(k)} \= \\Delta^k y\_i / k\!$ for orders $k \\in \\{1, 2, 3\\}$, providing direct linear representations of polynomial Taylor coefficients.  
   * Prepend two summary tokens, $\\mathbf{z}\_{\\text{affine}}$ and $\\mathbf{z}\_{\\text{geom}}$, to the encoder input, supervised via mean squared error to predict the sequence-wide linear slope $\\frac{\\text{Cov}(n, Y)}{\\text{Var}(n)}$ and geometric ratio $\\text{median}(y\_{i+1}/y\_i)$.  
   * Replace the 50 arbitrary Fourier moduli with Prime Fourier Embeddings (PFE) across the first 16 odd primes, isolating modular residues into orthogonal channels.

### **Phase 3: Long-Term Pre-Training Realignment and Policy Regularization**

The final objective is to permanently eliminate AST idiom memorization and establish scalable self-learning.

1. **Continuous Surrogate Reward Integration**:  
   * Replace binary execution rewards in early-to-mid GRPO training iterations with the continuous log-distance surrogate metric:  
     $$R\_{\\text{dense}}(P, Y) \= \\frac{1}{20} \\sum\_{n=0}^{19} \\frac{1}{1 \+ \\log\_{10}(|P(n) \- y\_n| \+ 1)}$$  
   * Apply this reward alongside an AST parsimony regularizer $-\\gamma \\cdot \\text{nodes}(P)$ to maintain non-zero advantage variance across sampling groups while penalizing unnecessarily complex program trees.  
2. **Synthetic Sweep Generation**:  
   * Restructure the procedural dataset generator to apply random affine scaling ($\\widetilde{Y} \= \\alpha Y \+ \\beta$) with $\\alpha \\sim \\pm 10^{\\mathcal{U}(0, 5)}$ to every generated program skeleton.  
   * This forces the model to bind multiplicative and additive constants dynamically from cross-attention representations, preventing the network from collapsing to default identity idioms and ensuring robust literal grounding across arbitrary integer domains.

