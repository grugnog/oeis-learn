# **Architecture and Theoretical Foundations of Tri-Stream Integer Encoding for Neuro-Symbolic OEIS Algorithm Synthesis**

## **Computational Challenges of Integer Sequences in Neural Architectures**

The Online Encyclopedia of Integer Sequences (OEIS) catalogues over 391,710 entries spanning combinatorics, number theory, graph theory, and algebra1. Modeling integer sequences in OEIS presents severe computational challenges for deep learning architectures1. Sequences in OEIS exhibit extreme numerical heterogeneity: values range from single-digit constants to astronomical factorials ($n\!$), double exponentials ($2^{2^n}$), and high-degree partition functions, resulting in dynamic ranges spanning hundreds of orders of magnitude within a single sequence1.  
Standard sequence modeling relies on tokenization algorithms such as Byte-Pair Encoding (BPE), WordPiece, or fixed-vocabulary integer mappings1. When applied to OEIS integer sequences, these discrete tokenization techniques fail due to three primary structural limitations1:

> 1. **Out-of-Vocabulary (OOV) Bound Breakdown**: Assigning discrete token IDs to integers restricts the model to a fixed vocabulary (e.g., numbers between $-10,000$ and $10,000$)1. Any unseen or large integer encountered during inference is mapped to a generic \[UNK\] token, completely erasing the numerical identity and scale of the sequence element1.  
> 2. **Fragmentation of Arithmetic Invariants**: Character- or digit-level tokenization decomposes large numbers into strings of isolated digit tokens (e.g., $123,456 \\to \\text{"123"}, \\text{"456"}$)3. While this avoids OOV tokens, it artificially multiplies sequence lengths, increases self-attention quadratic complexity $\\mathcal{O}(N^2)$, and obscures spatial and algebraic relations across positions3.  
> 3. **Opaque Representation of Modular and Multiplicative Structure**: Discrete token IDs treat numerically adjacent integers (e.g., $999$ and $1000$) or arithmetically related integers (e.g., $2^k$ and $2^{k+1}$) as orthogonal vectors in embedding space1. The model is forced to re-learn basic ring arithmetic and modulo operations end-to-end from sparse data, rather than exploiting structural homomorphisms inherent to number theory1.

Continuous numerical encodings, such as xVal6, map numbers to scalar multiples of a single vector embedding8. While xVal avoids OOV errors, it introduces severe failure modes when integrated into deep Transformer backbones9. Normalization layers—specifically LayerNorm ($\\text{LN}(\\mathbf{x}) \= \\frac{\\mathbf{x} \- \\mu}{\\sigma} \\odot \\boldsymbol{\\gamma} \+ \\boldsymbol{\\beta}$) and RMSNorm—rescale intermediate hidden activations, suppressing continuous magnitude information across deep layers3. Furthermore, simple continuous scalar encodings cannot capture periodic or modular arithmetic properties, causing models to perform poorly on sequences governed by modular recurrences (e.g., Fibonacci numbers modulo $m$)1.

| Integer Representation Strategy | Dynamic Value Range | Multiplicative / Modulo Structure | Normalization Stability | Token Expansion Overhead |
| :---- | :---- | :---- | :---- | :---- |
| **Discrete Subword Tokenization** | Unbounded (causes OOV) | Poor (treated as orthogonal IDs) | High | Minimal |
| **Digit-Wise Character Tokenization** | Unbounded | Moderate | High | Severe ($\\mathcal{O}(L \\cdot D)$ growth) |
| **xVal Continuous Scalar Encoding** | Bounded by floating point | Non-existent | Poor (squashed by LayerNorm) | Zero (1 token per number) |
| **Fourier Number Embeddings (FoNE)** | Bounded by phase resolution | High (periodic signals) | Moderate | Minimal |
| **IntSeqBERT Dual-Stream** | Unbounded ($\\mathbb{R}^{\\ge 0}$ log scale) | High (100 explicit moduli) | High (FiLM modulated) | Zero (1 position per integer) |

To resolve these structural bottlenecks, representation learning for integer sequences requires decoupling absolute numerical scale from internal algebraic structure1.

## **Deconstruction and Analysis of the IntSeqBERT Architecture**

The IntSeqBERT architecture proposed by Nakasho (2026) replaces tokenization with a dual-stream continuous embedding representation that explicitly separates growth magnitude from modular arithmetic structure1. Each integer $x\_i \\in \\mathbb{Z}$ in an OEIS sequence $\\mathbf{X} \= (x\_1, x\_2, \\dots, x\_N)$ is encoded along two axes and combined using Feature-wise Linear Modulation (FiLM)1.

### **Magnitude Stream Formalism**

The magnitude stream embeds the continuous growth scale of absolute integer values using a base-10 logarithmic transformation1. For a given integer $x\_i$, its continuous magnitude scalar $v\_i \\in \\mathbb{R}^{\\ge 0}$ is computed as:

$$v\_i \= \\begin{cases} 0 & \\text{if } x\_i \= 0 \\\\ 1 \+ \\log\_{10} \\vert{}x\_i\\vert{} & \\text{if } x\_i \\neq 0 \\end{cases}$$  
This logarithmic transformation compresses values spanning thousands of orders of magnitude into a stable, continuous linear range1. The continuous scalar $v\_i$ is subsequently projected into a $d$-dimensional magnitude embedding space using a parameterized multi-layer perceptron:

$$\\mathbf{E}\_{\\text{mag}}(x\_i) \= \\text{MLP}\_{\\text{mag}}(v\_i) \\in \\mathbb{R}^d$$

### **Modulo-Spectrum Stream Formalism**

To encode arithmetic and periodic structure independently of magnitude, IntSeqBERT computes modular residues across a spectrum of 100 moduli $\\mathcal{M} \= \\{2, 3, 4, \\dots, 101\\}$1. For each modulus $m \\in \\mathcal{M}$, the exact residue is calculated as $r\_i^{(m)} \= x\_i \\bmod m \\in \\{0, 1, \\dots, m-1\\}$1.  
To preserve the cyclic continuity of modular arithmetic ($\\mathbb{Z} / m\\mathbb{Z}$), each residue is mapped into a two-dimensional Fourier phase embedding1:

$$\\mathbf{\\phi}\_i^{(m)} \= \\left\[ \\sin\\left( \\frac{2\\pi (x\_i \\bmod m)}{m} \\right), \\, \\cos\\left( \\frac{2\\pi (x\_i \\bmod m)}{m} \\right) \\right\] \\in \\mathbb{R}^2$$  
The individual trigonometric vectors across all 100 moduli are concatenated into a unified residue spectrum vector $\\mathbf{\\Phi}\_i \\in \\mathbb{R}^{200}$:

$$\\mathbf{\\Phi}\_i \= \\bigoplus\_{m=2}^{101} \\mathbf{\\phi}\_i^{(m)}$$  
The unified spectrum vector is then passed through a linear transformation to produce the modular stream embedding $\\mathbf{E}\_{\\text{mod}}(x\_i) \\in \\mathbb{R}^d$:

$$\\mathbf{E}\_{\\text{mod}}(x\_i) \= \\mathbf{W}\_{\\text{mod}} \\mathbf{\\Phi}\_i \+ \\mathbf{b}\_{\\text{mod}}$$

### **Feature-wise Linear Modulation (FiLM) Fusion**

Rather than combining the magnitude and modular embeddings via simple addition or concatenation, IntSeqBERT fuses the streams using Feature-wise Linear Modulation (FiLM)1. The modular spectrum embedding generates scale ($\\boldsymbol{\\gamma}\_i$) and shift ($\\boldsymbol{\\beta}\_i$) conditioning vectors that modulate the magnitude embedding1:

$$\\boldsymbol{\\gamma}\_i \= \\mathbf{W}\_\\gamma \\mathbf{E}\_{\\text{mod}}(x\_i) \+ \\mathbf{b}\_\\gamma \\in \\mathbb{R}^d$$

$$\\boldsymbol{\\beta}\_i \= \\mathbf{W}\_\\beta \\mathbf{E}\_{\\text{mod}}(x\_i) \+ \\mathbf{b}\_\\beta \\in \\mathbb{R}^d$$

$$\\mathbf{h}\_i^{(0)} \= \\boldsymbol{\\gamma}\_i \\odot \\mathbf{E}\_{\\text{mag}}(x\_i) \+ \\boldsymbol{\\beta}\_i$$  
where $\\odot$ denotes the Hadamard (element-wise) product4. This conditioning mechanism enables the modular arithmetic stream to directly alter the feature activations of the growth magnitude stream before feeding the combined representation into the Transformer encoder layers1.

### **Multi-Task Objective and Prediction Heads**

During pre-training under a masked sequence modeling objective (mask probability $p \= 0.15$), three independent prediction heads extract structural properties for each masked position $i$1:

> * **Magnitude Regression Head**: Predicts continuous log-magnitude scalar $\\hat{v}\_i$ via Mean Squared Error (MSE) loss1:  
>   $$\\mathcal{L}\_{\\text{mag}} \= \\frac{1}{\\vert{}\\mathcal{M}\_{\\text{mask}}\\vert{}} \\sum\_{i \\in \\mathcal{M}\_{\\text{mask}}} (v\_i \- \\hat{v}\_i)^2$$  
> * **Sign Classification Head**: Predicts integer sign $s\_i \\in \\{+, \-, 0\\}$ as a 3-class categorical target using Cross-Entropy (CE) loss1:  
>   $$\\mathcal{L}\_{\\text{sign}} \= \-\\frac{1}{\\vert{}\\mathcal{M}\_{\\text{mask}}\\vert{}} \\sum\_{i \\in \\mathcal{M}\_{\\text{mask}}} \\sum\_{c \\in \\{+, \-, 0\\}} y\_{i, c} \\log \\hat{p}\_{i, c}$$  
> * **Modulo Spectrum Classification Head**: Computes 100 independent cross-entropy losses, one for each modulus $m \\in \\{2, \\dots, 101\\}$1:  
>   $$\\mathcal{L}\_{\\text{mod}} \= \\frac{1}{100} \\sum\_{m=2}^{101} \\left( \-\\frac{1}{\\vert{}\\mathcal{M}\_{\\text{mask}}\\vert{}} \\sum\_{i \\in \\mathcal{M}\_{\\text{mask}}} \\sum\_{k=0}^{m-1} \\mathbb{I}(r\_i^{(m)} \= k) \\log \\hat{p}\_{i, k}^{(m)} \\right)$$

The global loss function optimizes a weighted linear combination of these objectives1:

$$\\mathcal{L}\_{\\text{total}} \= \\lambda\_{\\text{mag}} \\mathcal{L}\_{\\text{mag}} \+ \\lambda\_{\\text{sign}} \\mathcal{L}\_{\\text{sign}} \+ \\lambda\_{\\text{mod}} \\mathcal{L}\_{\\text{mod}}$$

## **Modulo Spectrum Analysis and Probabilistic CRT Decoding**

A primary theoretical discovery established by Nakasho (2026) is the relationship between composite moduli, Euler's totient function $\\varphi(m)$, and Normalized Information Gain (NIG)1. For a target modulus $m$, Normalized Information Gain quantifies the reduction in entropy achieved by predicting $x\_i \\bmod m$ relative to the uniform prior over $\\mathbb{Z}/m\\mathbb{Z}$:

$$\\text{NIG}(m) \= \\frac{H\_{\\text{prior}}(m) \- H\_{\\text{model}}(m)}{H\_{\\text{prior}}(m)} \= 1 \- \\frac{-\\sum\_{k=0}^{m-1} \\hat{p}\_k^{(m)} \\log \\hat{p}\_k^{(m)}}{\\log m}$$  
Statistical evaluation across 274,705 OEIS sequences demonstrates a strong negative correlation between $\\text{NIG}(m)$ and Euler's totient ratio $\\frac{\\varphi(m)}{m}$ ($r \= \-0.851, p \< 10^{-28}$)1.  
This negative correlation provides empirical evidence that composite moduli aggregate arithmetic structure more efficiently than isolated prime moduli1. Euler's totient function $\\varphi(m) \= m \\prod\_{p \\vert{} m} (1 \- p^{-1})$ measures the number of integers up to $m$ that are coprime to $m$. A lower totient ratio $\\frac{\\varphi(m)}{m}$ indicates that $m$ shares prime factors with a larger fraction of integers1. Under the Chinese Remainder Theorem (CRT), the ring $\\mathbb{Z}/m\\mathbb{Z}$ decomposes into the direct product of prime-power rings5:

$$\\mathbb{Z} / m\\mathbb{Z} \\cong \\mathbb{Z} / p\_1^{k\_1}\\mathbb{Z} \\times \\mathbb{Z} / p\_2^{k\_2}\\mathbb{Z} \\times \\dots \\times \\mathbb{Z} / p\_l^{k\_l}\\mathbb{Z}$$  
Consequently, composite moduli with low totient ratios (e.g., $m \= 60, 84, 90$) simultaneously impose multiple prime-power constraints on the target position1. The Transformer attention mechanism captures these multi-scale congruence relationships directly from sequence context1.

| Modulus (m) | Modulus Type | Totient Ratio (mφ(m)​) | IntSeqBERT Accuracy (%) | Vanilla Baseline (%) | Ablation w/o Modulo Stream (%) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **2** | Prime | 0.5000 | 85.65 | 81.40 | 72.13 |
| **3** | Prime | 0.6667 | 72.62 | 65.22 | 53.72 |
| **5** | Prime | 0.8000 | 60.37 | 50.07 | 42.63 |
| **12** | Composite ($2^2 \\cdot 3$) | 0.3333 | 68.45 | 48.12 | 34.10 |
| **60** | Composite ($2^2 \\cdot 3 \\cdot 5$) | 0.2667 | 62.10 | 38.90 | 22.45 |
| **97** | Prime | 0.9897 | 41.20 | 28.50 | 14.20 |

To convert continuous magnitude predictions ($\\hat{v}\_i$), sign distributions ($\\hat{s}\_i$), and categorical modulo distributions ($\\{\\hat{r}\_i^{(m)}\\}$) back into concrete integer values, IntSeqBERT employs a Probabilistic CRT Solver1. The solver computes candidate bounds using the predicted magnitude $\\hat{v}\_i \\pm \\epsilon$, corresponding to an absolute dynamic search interval $\[N\_{\\min}, N\_{\\max}\]$ of width $\\Delta\_n$11.  
The probabilistic solver operates in three dynamic operational modes determined by the magnitude uncertainty $\\Delta\_n$11:

> * **Dense Mode ($\\Delta\_n \\le 10^6$)**: Directly evaluates all integers $x \\in \[N\_{\\min}, N\_{\\max}\]$, scoring each integer by summing log-likelihoods across the 100 predicted residue distributions11:  
>   $$\\text{Score}(x) \= \\log P(\\text{sign}(x)) \+ \\sum\_{m=2}^{101} \\log P(x \\bmod m \\mid m)$$  
> * **Sieve Mode ($10^6 \< \\Delta\_n \\le 10^{14}$)**: Filters the search space using a beam search over high-confidence moduli subsets11. Moduli with predicted residue entropy below a threshold $\\tau$ are selected to form a system of congruences $x \\equiv r\_m \\pmod m$, constructing candidate integers via standard CRT reconstruction before score evaluation11.  
> * **Sparse CRT Mode ($\\Delta\_n \> 10^{14}$)**: Bypasses bounded search completely11. The solver constructs integer candidates directly from the Cartesian product of top-$k$ residue predictions across prime moduli pairs using lattice basis reduction (LLL algorithm), ensuring numerical recovery for large values11.

Combining the dual-stream representations with the probabilistic CRT solver yields a 7.4-fold improvement in Top-1 next-term prediction accuracy over standard tokenized Transformer baselines (19.09% vs. 2.59%)1.

## **Architectural Synthesis of the Tri-Stream Encoder for oeis-learn**

While IntSeqBERT successfully captures global magnitude growth and modular periodicities1, algorithm synthesis in oeis-learn requires identifying precise local recurrence operations, higher-order finite differences, and $p$-adic factorizations. To satisfy these requirements, the dual-stream design is extended into a **Tri-Stream Encoder Architecture**.

### **Stream 1: Continuous Log-Magnitude and Growth Dynamics Stream ($\\mathbf{S}\_1$)**

The first stream encodes scale using a signed logarithmic transformation:

$$v\_i \= \\text{sign}(x\_i) \\cdot \\left( 1 \+ \\log\_{10}(\\vert{}x\_i\\vert{} \+ 1\) \\right) \\in \\mathbb{R}$$  
This scalar is projected via a 2-layer MLP with GELU activations to produce the continuous scale vector $\\mathbf{S}\_{1, i} \\in \\mathbb{R}^d$.

### **Stream 2: Modulo-Spectrum Spectral Stream ($\\mathbf{S}\_2$)**

The second stream retains the 100-moduli trigonometric phase representation ($m \\in \\{2, \\dots, 101\\}$)1. The concatenated 200-dimensional phase vector $\\mathbf{\\Phi}\_i$ is transformed through a linear embedding layer to yield the modular state $\\mathbf{S}\_{2, i} \\in \\mathbb{R}^d$.

### **Stream 3: Local Difference and $p$-Adic Valuation Stream ($\\mathbf{S}\_3$)**

The third stream captures relative step dynamics and divisibility patterns that are suppressed in log-scale transformations4. For sequence element $x\_i$, Stream 3 computes:

> 1. **First-Order Finite Difference**: $\\Delta x\_i \= x\_i \- x\_{i-1}$  
> 2. **Second-Order Finite Difference**: $\\Delta^2 x\_i \= \\Delta x\_i \- \\Delta x\_{i-1} \= x\_i \- 2x\_{i-1} \+ x\_{i-2}$  
> 3. **$p$-Adic Valuations**: $v\_p(x\_i) \= \\max \\{ k \\in \\mathbb{N}\_0 : p^k \\mid x\_i \\}$ for small primes $p \\in \\{2, 3, 5, 7, 11, 13\\}$, capped at $k\_{\\max} \= 16$.

The finite differences are logarithmically transformed:

$$d\_i^{(1)} \= \\text{sign}(\\Delta x\_i) \\cdot \\left( 1 \+ \\log\_{10}(\\vert{}\\Delta x\_i\\vert{} \+ 1\) \\right)$$

$$d\_i^{(2)} \= \\text{sign}(\\Delta^2 x\_i) \\cdot \\left( 1 \+ \\log\_{10}(\\vert{}\\Delta^2 x\_i\\vert{} \+ 1\) \\right)$$  
The $p$-adic valuation exponents are mapped to learned ordinal embeddings $\\mathbf{E}\_{\\text{padic}}(x\_i) \\in \\mathbb{R}^{d\_p}$. The components are concatenated and projected to construct the local dynamic embedding $\\mathbf{S}\_{3, i} \\in \\mathbb{R}^d$:

$$\\mathbf{S}\_{3, i} \= \\mathbf{W}\_d \\left\[ d\_i^{(1)} \\,\\Vert{}\\, d\_i^{(2)} \\,\\Vert{}\\, \\mathbf{E}\_{\\text{padic}}(x\_i) \\right\] \+ \\mathbf{b}\_d$$

### **Hierarchical Tri-Stream FiLM Fusion**

The three streams are integrated using a two-stage hierarchical FiLM block:

> * **Stage 1 Modulo-Scale Modulation**:  
>   $$\\boldsymbol{\\gamma}\_i^{(1)}, \\boldsymbol{\\beta}\_i^{(1)} \= \\text{Split}\\left(\\mathbf{W}\_{\\text{FiLM1}} \\mathbf{S}\_{2, i} \+ \\mathbf{b}\_{\\text{FiLM1}}\\right)$$  
>   $$\\mathbf{H}\_{12, i} \= \\boldsymbol{\\gamma}\_i^{(1)} \\odot \\mathbf{S}\_{1, i} \+ \\boldsymbol{\\beta}\_i^{(1)}$$  
> * **Stage 2 Local Dynamic Modulation**:  
>   $$\\boldsymbol{\\gamma}\_i^{(2)}, \\boldsymbol{\\beta}\_i^{(2)} \= \\text{Split}\\left(\\mathbf{W}\_{\\text{FiLM2}} \\mathbf{S}\_{3, i} \+ \\mathbf{b}\_{\\text{FiLM2}}\\right)$$  
>   $$\\mathbf{Z}\_i \= \\boldsymbol{\\gamma}\_i^{(2)} \\odot \\mathbf{H}\_{12, i} \+ \\boldsymbol{\\beta}\_i^{(2)}$$

The resulting unified embedding $\\mathbf{Z}\_i \\in \\mathbb{R}^d$ combines continuous magnitude scale, periodic modular congruences, and exact local difference dynamics prior to processing by the Transformer backbone.

| Hyperparameter / Module | IntSeqBERT Baseline | oeis-learn Tri-Stream Architecture | Structural Function |
| :---- | :---- | :---- | :---- |
| **Model Scale (Parameters)** | 91.5M (Large)1 | 142.8M (Extended) | Scaled capacity for program synthesis |
| **Input Streams** | Dual (Magnitude, Modulo)1 | Tri (Magnitude, Modulo, Difference/$p$-adic) | Multi-axis number characterization |
| **Moduli Range** | $m \\in \\{2, \\dots, 101\\}$ \[cite: 1\] | $m \\in \\{2, \\dots, 101\\}$ \[cite: 1\] | Modular spectrum coverage |
| **Trigonometric Embedding** | Sin/Cos pair per modulus1 | Sin/Cos pair per modulus1 | Continuous cyclic phase encoding |
| **Local Differences** | None | $\\Delta x\_i, \\Delta^2 x\_i$ \+ $p$-adic ($p \\le 13$) | Local step and divisibility tracking |
| **Fusion Mechanism** | Single-step FiLM1 | Hierarchical Two-Stage FiLM | Sequential feature conditioning |
| **Encoder Backbone** | 12-layer Bidirectional Transformer | 16-layer Bidirectional Transformer | Deep context feature extraction |
| **Numerical Precision** | FP32 (No AMP)11 | FP32 (No AMP)11 | Prevents gradient underflows in phase functions |

## **Neuro-Symbolic Integration and Algorithmic Program Synthesis**

The primary objective of oeis-learn is to perform symbolic algorithm synthesis—mapping an observed integer sequence to its exact generating formula, recurrence relation, or algorithm expressed in a domain-specific language (DSL). The Tri-Stream Encoder functions as the perception backbone within a dual-path neuro-symbolic framework.

### **Synthesis Pipeline Mechanics**

The neuro-symbolic synthesis process proceeds through four sequential stages:

> 1. **Latent Context Extraction**: Given a partial integer sequence $\\mathbf{X}\_{1:N}$, the Tri-Stream Encoder computes unified sequence representations $\\mathbf{Z}\_{1:N} \\in \\mathbb{R}^{N \\times d}$.  
> 2. **Neural Candidate Generation**: The prediction heads and Probabilistic CRT Solver generate concrete candidate predictions $\\hat{x}\_{N+1}, \\hat{x}\_{N+2}, \\dots, \\hat{x}\_{N+k}$ for future sequence terms1.  
> 3. **Symbolic AST Decoding**: An autoregressive Transformer decoder conditions on $\\mathbf{Z}\_{1:N}$ via cross-attention to synthesize an Abstract Syntax Tree (AST) representing the generating algorithm. The DSL grammar contains standard arithmetic operations ($+, \-, \\times, \\div, \\bmod$), functional primitives ($\\text{gcd}, \\text{lcm}, \\text{factorial}, \\text{fibonacci}$), and sequence memory registers ($x\_{n-1}, x\_{n-2}$).  
> 4. **Execution-Guided Search Verification**: Synthesized AST programs are evaluated on input indices $n=1 \\dots N$. Programs producing outputs that deviate from the ground-truth sequence or CRT Solver estimates are pruned.

By combining neural candidate predictions with symbolic AST execution, oeis-learn restricts the program search space, addressing the combinatorial explosion that affects pure symbolic search methods15.

## **Implementation Specifications and Engineering Roadmap**

The dual-stream architecture of IntSeqBERT establishes that continuous log-magnitude transformations combined with modular phase embeddings overcome the out-of-vocabulary and gradient instability issues characteristic of discrete tokenized models1. The strong negative correlation ($r \= \-0.851$) between Euler's totient ratio $\\frac{\\varphi(m)}{m}$ and Normalized Information Gain confirms that composite moduli capture arithmetic invariants effectively through Chinese Remainder Theorem ring decompositions1.  
Building on these empirical results, the Tri-Stream Encoder design for oeis-learn adds a finite-difference and $p$-adic valuation stream, providing the local dynamic features required for symbolic program synthesis.  
To maintain numerical stability and training efficiency, implementation should follow these structural parameters11:

> * **Strict FP32 Execution**: Automatic Mixed Precision (AMP / FP16 / BF16) must be disabled11. Trigonometric phase functions ($\\sin, \\cos$) for high moduli ($m \\approx 100$) and continuous log-space transformations produce small gradient updates that underflow under 16-bit floating-point representations11.  
> * **Optimization Protocol**: Models should be trained using AdamW with weight decay $\\lambda \= 0.01$, an initial learning rate $\\eta \= 5 \\times 10^{-5}$, and a 10% linear warmup fraction over 200 epochs without early stopping11.  
> * **Batch Sizing and Hardware Gradient Accumulation**: Training should employ a physical batch size of 32 per GPU with 2 gradient accumulation steps (effective batch size of 64\)11.  
> * **Target Head Weighting**: Loss weights should be set to $\\lambda\_{\\text{mag}} \= 1.0$, $\\lambda\_{\\text{sign}} \= 0.5$, and $\\lambda\_{\\text{mod}} \= 2.0$ to ensure balanced gradient propagation across regression and cross-entropy prediction heads1.

#### **Works cited**

> 1. IntSeqBERT: Learning Arithmetic Structure in OEIS via Modulo, [https://arxiv.org/html/2603.05556v1](https://arxiv.org/html/2603.05556v1)  
> 2. CiteN \- OeisWiki, [https://oeis.org/wiki/CiteN](https://oeis.org/wiki/CiteN)  
> 3. (PDF) FoNE: Precise Single-Token Number Embeddings via Fourier, [https://www.researchgate.net/publication/389056301\_FoNE\_Precise\_Single-Token\_Number\_Embeddings\_via\_Fourier\_Features](https://www.researchgate.net/publication/389056301_FoNE_Precise_Single-Token_Number_Embeddings_via_Fourier_Features)  
> 4. Efficient numeracy in language models through single-token number, [https://arxiv.org/html/2510.06824v1](https://arxiv.org/html/2510.06824v1)  
> 5. IntSeqBERT: Learning Arithmetic Structure in OEIS via Modulo, [https://arxiv.org/pdf/2603.05556](https://arxiv.org/pdf/2603.05556)  
> 6. xVal: A Continuous Number Encoding for Large Language Models, [https://openreview.net/pdf/472ce83a3cb5f9c7cee6ed7dd7ad0ae1cb7d20ae.pdf](https://openreview.net/pdf/472ce83a3cb5f9c7cee6ed7dd7ad0ae1cb7d20ae.pdf)  
> 7. A Continuous Numerical Tokenization for Scientific Language Models, [https://arxiv.org/html/2310.02989v2](https://arxiv.org/html/2310.02989v2)  
> 8. xVal: A continuous number encoding for large language models, [https://news.ycombinator.com/item?id=37936005](https://news.ycombinator.com/item?id=37936005)  
> 9. DriveCode: Domain Specific Numerical Encoding for LLM-Based, [https://arxiv.org/html/2603.00919v2](https://arxiv.org/html/2603.00919v2)  
> 10. Efficient numeracy in language models through single-token number, [https://www.alphaxiv.org/abs/2510.06824](https://www.alphaxiv.org/abs/2510.06824)  
> 11. IntSeqBERT: Learning Arithmetic Structure in OEIS via Modulo, [https://arxiv.org/html/2603.05556v2](https://arxiv.org/html/2603.05556v2)  
> 12. IntSeqBERT: Learning Arithmetic Structure in OEIS via Modulo, [https://chatpaper.com/ja/paper/249423](https://chatpaper.com/ja/paper/249423)  
> 13. FiLM: Visual Reasoning with a General Conditioning Layer \- arXiv, [https://arxiv.org/abs/1709.07871](https://arxiv.org/abs/1709.07871)  
> 14. IntSeqBERT: Learning Arithmetic Structure in OEIS via Modulo, [https://arxiv.org/abs/2603.05556](https://arxiv.org/abs/2603.05556)  
> 15. Generating conjectures on fundamental constants with the, [https://www.researchgate.net/publication/349013723\_Generating\_conjectures\_on\_fundamental\_constants\_with\_the\_Ramanujan\_Machine](https://www.researchgate.net/publication/349013723_Generating_conjectures_on_fundamental_constants_with_the_Ramanujan_Machine)