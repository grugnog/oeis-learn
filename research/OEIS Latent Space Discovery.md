# **Latent Space Mapping and Automated Discovery of Mathematical Relations in the OEIS**

Automating mathematical discovery across the On-Line Encyclopedia of Integer Sequences (OEIS) requires mapping discrete numerical sequences into a continuous, geometrically structured latent space. By converting over 350,000 integer sequences into 768-dimensional latent representations, deep learning models can uncover hidden structural similarities, functional equivalences, and algebraic transformations that transcend manual tagging or simple rule-based matching1. Achieving this objective necessitates a rigorous combination of self-supervised learning, high-dimensional manifold analysis, and multi-stage symbolic verification pipelines4.

## **Self-Supervised Contrastive Learning Architecture for Mathematical Sequence Embeddings**

To construct a meaningful latent space for mathematical sequences, the encoder must learn representations that reflect functional and structural relationships rather than trivial surface-level numerical overlaps1. Standard supervised training is inapplicable due to the absence of exhaustive class labels for 350,000 sequences2. Consequently, self-supervised representation learning objectives must be tailored to the structural dynamics of discrete mathematics4.

### **Algebraic Transformation Algebra for Positive Pair Generation**

In computer vision or natural language processing, positive pairs are generated using stochastic augmentations such as cropping, rotation, or token masking9. In discrete mathematics, random noise destroys exact algebraic identities. Instead, positive pairs must be generated using deterministic functional operators that preserve underlying structural invariants4.  
Given a sequence $A \= (a\_n)\_{n=0}^{\\infty}$, a positive pair $(A, A^+)$ is constructed by applying exact mathematical operations:

> * **Partial Sum Operator ($\\mathcal{S}$)**: Maps $a\_n$ to $s\_n \= \\sum\_{k=0}^n a\_k$. This operation introduces an integral-like relationship, linking sequences through rate-of-growth transformations.  
> * **First Difference Operator ($\\Delta$)**: Maps $a\_n$ to $\\Delta a\_n \= a\_n \- a\_{n-1}$ (for $n \\ge 1$). This acts as a discrete derivative, positioning sequence embeddings along a continuum of differential orders.  
> * **Binomial Transform Operator ($\\mathcal{B}$)**: Maps $a\_n$ to $b\_n \= \\sum\_{k=0}^n \\binom{n}{k} (-1)^{n-k} a\_k$ or its inverse $b\_n \= \\sum\_{k=0}^n \\binom{n}{k} a\_k$. The binomial transform alters the ordinary generating function $A(x)$ to $B(x) \= \\frac{1}{1-x} A\\left(\\frac{x}{1-x}\\right)$, mapping exponential and polynomial sequence families into equivalent functional clusters.  
> * **Euler Transform Operator ($\\mathcal{E}$)**: Connects integer partitions and multiset constructions, transforming generating functions according to $1 \+ E(x) \= \\prod\_{k=1}^{\\infty} (1 \- x^k)^{-a\_k}$.  
> * **Shift Operator ($\\mathcal{T}\_k$)**: Maps $a\_n$ to $a\_{n+k}$, forcing the encoder to learn shift-invariant representations that ignore initial boundary conditions or offset indices.  
> * **Sub-sampling / Decimation Operator ($\\mathcal{D}\_k$)**: Maps $a\_n$ to $a\_{kn}$, capturing periodic sub-structures and multisection properties of linear recurrences.

By treating a sequence $A$ and its transformed counterpart $T(A)$ as positive views, the encoder is forced to map derivative, integral, and transform pairs to nearby points or along predictable directional vectors in the latent space1.

### **Loss Function Formulation and Representation Collapse Mitigation**

To train the 768-dimensional latent space without explicit class labels, the optimization objective must prevent representation collapse—a failure mode where all sequences map to a single constant vector11.

#### **Contrastive InfoNCE / NT-Xent Objective**

The Normalized Temperature-Scaled Cross-Entropy (NT-Xent) loss utilizes explicit negative sampling within mini-batches11. For a mini-batch of $N$ sequences yielding $2N$ augmented views, the loss for a positive pair $(i, j)$ is defined as:

$$\\mathcal{L}\_{\\text{NT-Xent}}^{(i,j)} \= \-\\log \\frac{\\exp\\left(\\text{sim}(z\_i, z\_j) / \\tau\\right)}{\\sum\_{k=1}^{2N} \\mathbb{I}\_{\[k \\neq i\]} \\exp\\left(\\text{sim}(z\_i, z\_k) / \\tau\\right)}$$  
where $\\text{sim}(u, v) \= \\frac{u^T v}{\\Vert{}u\\Vert{}\_2 \\Vert{}v\\Vert{}\_2}$ represents cosine similarity, $\\tau$ is a temperature hyperparameter controlling sensitivity to hard negatives, and $\\mathbb{I}$ is an indicator function9.  
While NT-Xent is effective in traditional domain tasks, it suffers from the class collision problem in pure mathematics15. Two randomly selected sequences in a mini-batch sampled as "negatives" may actually share a deep, undiscovered identity3. Penalizing their similarity distorts the latent topology15.

#### **Covariance Regularization via Barlow Twins and VICReg**

To overcome class collision, non-contrastive objectives rely on decorrelation and variance constraints across batch features rather than explicit negative pairs12.  
The Barlow Twins loss operates directly on the cross-correlation matrix $C \\in \\mathbb{R}^{d \\times d}$ computed between the projected embedding outputs $Z\_a, Z\_b \\in \\mathbb{R}^{N \\times d}$ of two transformed views across a batch of size $N$13:

$$C\_{ik} \= \\frac{\\sum\_{b=1}^N z\_{b,i}^a z\_{b,k}^b}{\\sqrt{\\sum\_{b=1}^N (z\_{b,i}^a)^2} \\sqrt{\\sum\_{b=1}^N (z\_{b,k}^b)^2}}$$

$$\\mathcal{L}\_{\\text{Barlow Twins}} \= \\sum\_{i=1}^d (1 \- C\_{ii})^2 \+ \\lambda \\sum\_{i=1}^d \\sum\_{j \\neq i}^d C\_{ij}^2$$  
The diagonal invariance term forces the representation to be robust to algebraic transformations, while the off-diagonal redundancy reduction term decorrelates feature dimensions, maximizing vector capacity13.  
VICReg (Variance-Invariance-Covariance Regularization) expands this principle by explicitly decomposing the loss into three distinct geometric objectives12:

$$\\mathcal{L}\_{\\text{VICReg}} \= \\lambda s(Z\_a, Z\_b) \+ \\mu \\left\[ v(Z\_a) \+ v(Z\_b) \\right\] \+ \\nu \\left\[ c(Z\_a) \+ c(Z\_b) \\right\]$$

> 1. **Invariance Term $s(Z\_a, Z\_b)$**: Mean squared error between positive projection vectors, pulling transformed sequences together12:  
>    $$s(Z\_a, Z\_b) \= \\frac{1}{N} \\sum\_{i=1}^N \\Vert{}z\_{i}^a \- z\_{i}^b\\Vert{}\_2^2$$  
> 2. **Variance Term $v(Z)$**: Hinge loss on the standard deviation of each embedding dimension across the batch, preventing complete collapse by maintaining feature spread above a threshold $\\gamma$ (typically $\\gamma \= 1$)12:  
>    $$v(Z) \= \\frac{1}{d} \\sum\_{j=1}^d \\max\\left(0, \\gamma \- \\sqrt{\\text{Var}(Z\_{:,j}) \+ \\epsilon}\\right)$$  
> 3. **Covariance Term $c(Z)$**: Sum of squared off-diagonal elements of the covariance matrix $C(Z)$, preventing feature redundancy and dimension collapse12:  
>    $$c(Z) \= \\frac{1}{d} \\sum\_{i \\neq j} \\left\[ C(Z) \\right\]\_{ij}^2, \\quad \\text{where } C(Z) \= \\frac{1}{N \- 1} \\sum\_{i=1}^N (z\_i \- \\bar{z})(z\_i \- \\bar{z})^T$$

For oeis-learn, VICReg provides the optimal training objective12. By eliminating negative sampling, it avoids penalizing unidentified mathematical equivalences while guaranteeing that all 768 dimensions encode orthogonal, highly expressive mathematical features12.

## **Scalable Manifold Reduction and High-Dimensional Clustering**

Analyzing 350,000 sequence embeddings in a 768-dimensional space requires computational algorithms capable of capturing nonlinear manifold structures without succumbing to the curse of dimensionality7.

### **Topology Preservation via GPU-Accelerated UMAP**

Uniform Manifold Approximation and Projection (UMAP) models the high-dimensional vector space as a fuzzy simplicial set and constructs a low-dimensional layout that minimizes the fuzzy set cross-entropy between representation spaces7.  
For 350,000 768-dimensional vectors, UMAP parameterization requires exact configuration:

> * n\_neighbors \= 30: Balances local algebraic details (e.g., immediate transform pairs) with global mathematical structures (e.g., polynomial vs. exponential sequence families).  
> * min\_dist \= 0.05: Allows tight clustering of structurally identical sequences while maintaining smooth manifold boundaries between distinct mathematical classes.  
> * metric \= 'cosine': Alignment in latent space is driven primarily by vector direction rather than magnitude, matching the normalization used during VICReg training11.

### **Hierarchical Density Estimation via HDBSCAN**

Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) identifies clusters of varying densities without requiring a predefined cluster count $k$7. HDBSCAN operates by transforming the metric space based on mutual reachability distance:

$$d\_{\\text{mreach}, k}(a, b) \= \\max \\left\\{ \\text{core}\_k(a), \\text{core}\_k(b), d(a, b) \\right\\}$$  
where $\\text{core}\_k(x)$ is the distance from $x$ to its $k$-th nearest neighbor19.  
Sequences assigned noise labels (-1) by HDBSCAN represent mathematical primitives that do not belong to standard clusters (e.g., prime numbers, fundamental constants, or isolated exotic recurrences)3. Rather than discarding these noise points, they are flagged as candidates for unique mathematical properties or unclassified sequence classes3.

### **Computational Optimization and Complexity Analysis**

Processing 350,000 high-dimensional vectors on CPU architectures presents severe bottlenecks, as pairwise distance computations scale quadratically $O(N^2)$. Utilizing GPU acceleration via RAPIDS cuML accelerates UMAP by 25x–60x and HDBSCAN by up to 200x on modern GPU hardware (e.g., NVIDIA A10/A100), reducing runtimes from days to minutes7.

| Algorithm | Time Complexity | Memory Complexity | Preserved Topology | Primary Role in OEIS Discovery Pipeline |
| :---- | :---- | :---- | :---- | :---- |
| **PCA (Baseline)** | $O(d^2 N \+ d^3)$ | $O(N d)$ | Global Linear | Initial variance baseline & linear dimensionality reduction. |
| **t-SNE** | $O(N \\log N)$ | $O(N^2)$ | Local Only | Visualizing local sub-clusters; computationally prohibitive at scale. |
| **GPU-cuML UMAP** | $O(N \\log N)$ | $O(N \\cdot k)$ | Local & Global | Non-linear manifold reduction (768D to 2D/3D)7. |
| **GPU-cuML HDBSCAN** | $O(N \\log N)$ | $O(N \\cdot k)$ | Hierarchical Density | Unsupervised cluster assignment & outlier detection7. |
| **K-Means (Baseline)** | $O(N \\cdot k \\cdot d \\cdot i)$ | $O(N d)$ | Spherical Euclidean | Fast baseline; fails on non-spherical mathematical manifolds. |

## **Vector Arithmetic and Multi-Stage Theorem Verification Pipeline**

A structured latent space allows linear vector arithmetic to mirror abstract algebraic operations1. For example, given sequence representations $\\vec{v}(A)$, $\\vec{v}(B)$, and $\\vec{v}(C)$, relationships such as $\\vec{v}(A) \+ \\vec{v}(B) \\approx \\vec{v}(C)$ suggest candidate operational conjectures (e.g., term-wise addition, Cauchy convolution, or Dirichlet convolution)5. Similarly, fixed displacement vectors $\\vec{v}\_{\\Delta} \= \\vec{v}(T(A)) \- \\vec{v}(A)$ define transformation vectors in the latent manifold1.

### **Multi-Stage Verification Engine**

To distinguish geometric coincidences from genuine, unproven mathematical theorems, candidate relations identified via vector arithmetic must pass through a 4-stage automated verification pipeline5.

#### **Stage 1: Candidate Retrieval via Geometric Search**

For a given anchor sequence $A$, nearest-neighbor candidate tuples $(A, B, C)$ are extracted using Hierarchical Navigable Small World (HNSW) indexing under the geometric condition $\\Vert{}\\vec{v}(A) \+ \\vec{v}(B) \- \\vec{v}(C)\\Vert{}\_2 \< \\epsilon\_{\\text{geom}}$1.

#### **Stage 2: High-Precision Numerical Sampling**

Once candidate sequences are flagged, their exact terms are computed up to $N \= 1000$ terms using arbitrary-precision arithmetic via mpmath22. Sequences are evaluated as terms of sequence vectors or sampled value points of their associated ordinary generating functions $A(x) \= \\sum a\_n x^n$ evaluated at transcendental sample points $x\_0 \\in \\mathbb{R}$ within the radius of convergence23.

#### **Stage 3: Integer Relation Detection via the PSLQ Algorithm**

The PSLQ (Partial Sum of Least Squares) algorithm is the standard for discovering integer relations among real numbers8. Given a vector of $n$ high-precision numerical values $x \= (x\_1, x\_2, \\dots, x\_n) \\in \\mathbb{R}^n$, PSLQ searches for a vector of integers $a \= (a\_1, a\_2, \\dots, a\_n) \\in \\mathbb{Z}^n$, not all zero, such that8:

$$\\sum\_{i=1}^n a\_i x\_i \= 0$$  
PSLQ provides two critical mathematical guarantees during execution8:

> 1. **Lower Norm Bound $M$**: PSLQ computes a lower bound $M$ such that no integer relation vector $a$ exists with Euclidean norm $\\Vert{}a\\Vert{}\_2 \< M$8. If $M$ exceeds precision limits without finding a relation, the geometric vector candidate is rejected as a numerical artifact8.  
> 2. **Detection Threshold and Confidence Ratio**: When an integer relation is detected, the smallest entry in the internal reduced vector $y$ drops dramatically below the threshold $\\epsilon$8. The confidence ratio $\\frac{\\min\_i \\vert{}y\_i\\vert{}}{\\max\_i \\vert{}y\_i\\vert{}}$ drops from $\\sim 10^{-10}$ to $\< 10^{-100}$ (depending on the working precision digits used)8. A sharp drop confirms a true numerical identity with vanishingly small false-positive probability ($P \< 10^{-50}$)8.

#### **Stage 4: Formal Symbolic Proof Execution**

Verified PSLQ identities are passed to symbolic computer algebra systems (SageMath and SymPy) to generate formal algebraic proofs22.

> * **Recurrence Relation Solving**: The pipeline uses sympy.rsolve() to solve linear recurrence equations with constant or polynomial coefficients, deriving exact closed-form expressions for sequence terms22.  
> * **Generating Function Identities**: Equality between ordinary generating functions $A(x) \+ B(x) \= C(x)$ or exponential generating functions is verified algebraically by coefficient extraction or differential equation verification27.  
> * **Wronskian Telescoping Proofs**: For deeper identities involving second-order difference equations, exact proofs are constructed via Wronskian telescoping and Abel summation techniques, establishing exact limits27.  
> * **Hypergraph Integration**: Inspired by the Ramanujan Machine framework, verified identities are added as directed hyperedges within a unified OEIS hypergraph, where sequences act as nodes and verified transformations represent structural connections3.

| Pipeline Stage | Subsystem Component | Tools & Algorithms | Input Data | Failure / Verification Criteria |
| :---- | :---- | :---- | :---- | :---- |
| **1\. Candidate Retrieval** | Geometric Discovery | HNSW / Cosine Distance | 768D Latent Embeddings | Candidate triples $(A, B, C)$ with $\\Vert{}\\vec{v}\_A \+ \\vec{v}\_B \- \\vec{v}\_C\\Vert{} \< \\epsilon$1. |
| **2\. Numerical Sampling** | Arbitrary Precision Engine | mpmath / Python | Candidate OEIS Sequences | 100–1000 term evaluations ($\>500$ digit precision)22. |
| **3\. Relation Detection** | Integer Relation Search | PSLQ Algorithm | Precision Value Vectors | Integer vector $a \\in \\mathbb{Z}^k$; Confidence ratio drop $\< 10^{-50}$8. |
| **4\. Symbolic Proof** | Computer Algebra System | SageMath / SymPy | Candidate Identity Equation | Symbolic proof (rsolve, generating function, or differential identity)22. |

## **Strategic Synthesis and Future Outlook**

The integration of self-supervised representation learning, GPU-accelerated density clustering, and high-precision integer relation filtering creates a systematic methodology for automated mathematical discovery across the OEIS3.  
By structuring the latent space with non-contrastive VICReg objectives, the encoder models operational invariants such as partial sums, binomial transforms, and finite differences without incurring class-collision penalties12. Downstream GPU-accelerated algorithms (cuML UMAP and HDBSCAN) allow real-time exploration of the 350,000-sequence manifold, grouping unannotated sequences into structural families and flagging mathematical anomalies7. Finally, coupling geometric vector arithmetic with PSLQ integer relation detection and symbolic proof engines converts approximate latent space symmetries into rigorous, machine-verified mathematical theorems8. This architecture provides a scalable roadmap for unifying discrete sequence mathematics into a continuous, searchable hypergraph of mathematical knowledge3.

#### **Works cited**

> 1. An introduction to contrastive techniques for representation learning, [https://pablomirallesg.com/blog/contrastive-learning](https://pablomirallesg.com/blog/contrastive-learning)  
> 2. Contrastive Representation Learning \- Shairoz Sohail \- Medium, [https://shairozsohail.medium.com/contrastive-representation-learning-a-comprehensive-guide-part-1-foundations-90c1944dbd1e](https://shairozsohail.medium.com/contrastive-representation-learning-a-comprehensive-guide-part-1-foundations-90c1944dbd1e)  
> 3. A paper by the Ramanujan Machine group from the Technion, [https://www.technion.ac.il/en/blog/article/a-paper-by-the-ramanujan-machine-group-from-the-technion-presents-deep-connections-between-different-mathematical-formulas-for-the-constant-%CF%80/](https://www.technion.ac.il/en/blog/article/a-paper-by-the-ramanujan-machine-group-from-the-technion-presents-deep-connections-between-different-mathematical-formulas-for-the-constant-%CF%80/)  
> 4. Learning Representations Through Contrastive Neural Model, [https://openreview.net/forum?id=RE7hugbL6U](https://openreview.net/forum?id=RE7hugbL6U)  
> 5. Algorithm-assisted discovery of an intrinsic order among ... \- PNAS, [https://www.pnas.org/doi/10.1073/pnas.2321440121](https://www.pnas.org/doi/10.1073/pnas.2321440121)  
> 6. Publications \- The Ramanujan Machine, [https://ramanujanmachine.com/publications/](https://ramanujanmachine.com/publications/)  
> 7. Supercharging Machine Learning in Snowflake with NVIDIA CUDA, [https://www.snowflake.com/en/blog/nvidia-gpu-acceleration/](https://www.snowflake.com/en/blog/nvidia-gpu-acceleration/)  
> 8. The PSLQ Integer Relation Algorithm \- CECM, SFU, [https://www.cecm.sfu.ca/organics/papers/bailey/paper/html/node3.html](https://www.cecm.sfu.ca/organics/papers/bailey/paper/html/node3.html)  
> 9. Contrastive Representation Learning | Lil'Log, [https://lilianweng.github.io/posts/2021-05-31-contrastive/](https://lilianweng.github.io/posts/2021-05-31-contrastive/)  
> 10. Can Contrastive Learning Refine Embeddings \- arXiv, [https://arxiv.org/html/2404.08701v1](https://arxiv.org/html/2404.08701v1)  
> 11. Contrastive Learning Loss: NT-Xent & InfoNCE \- Medium, [https://medium.com/self-supervised-learning/nt-xent-loss-normalized-temperature-scaled-cross-entropy-loss-ea5a1ede7c40](https://medium.com/self-supervised-learning/nt-xent-loss-normalized-temperature-scaled-cross-entropy-loss-ea5a1ede7c40)  
> 12. VICReg: Variance-Invariance-Covariance Regularization for Self, [https://www.assemblyai.com/blog/review-vicreg-variance-invariance-covariance-regularization-for-self-supervised-learning](https://www.assemblyai.com/blog/review-vicreg-variance-invariance-covariance-regularization-for-self-supervised-learning)  
> 13. Barlow Twins: Self-Supervised Learning \- Encord, [https://encord.com/blog/barlow-twins-self-supervised-learning/](https://encord.com/blog/barlow-twins-self-supervised-learning/)  
> 14. Contrastive learning / self-supervised learning \- lecture-10 \- GitHub, [https://github.com/oseledets/dl2023/blob/main/lectures/lecture-10/lecture-10.ipynb](https://github.com/oseledets/dl2023/blob/main/lectures/lecture-10/lecture-10.ipynb)  
> 15. Contrastive vs Non-Contrastive SSL \- Emergent Mind, [https://www.emergentmind.com/topics/contrastive-and-non-contrastive-self-supervised-learning](https://www.emergentmind.com/topics/contrastive-and-non-contrastive-self-supervised-learning)  
> 16. Contrastive self-supervised representation learning without negative, [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1225312/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1225312/full)  
> 17. Evaluation of Barlow Twins and VICReg self-supervised learning for, [https://arxiv.org/html/2312.11240v1](https://arxiv.org/html/2312.11240v1)  
> 18. Even Faster and More Scalable UMAP on the GPU with NVIDIA cuML, [https://developer.nvidia.com/blog/even-faster-and-more-scalable-umap-on-the-gpu-with-rapids-cuml/](https://developer.nvidia.com/blog/even-faster-and-more-scalable-umap-on-the-gpu-with-rapids-cuml/)  
> 19. Frequently Asked Questions — umap 0.5.8 documentation, [https://umap-learn.readthedocs.io/en/latest/faq.html](https://umap-learn.readthedocs.io/en/latest/faq.html)  
> 20. Benchmarks — NVIDIA cuML, [https://docs.rapids.ai/api/cuml/stable/cuml-accel/benchmarks/](https://docs.rapids.ai/api/cuml/stable/cuml-accel/benchmarks/)  
> 21. The Ramanujan Machine discovers a new mathematical structure, [https://www.reddit.com/r/math/comments/16003f7/the\_ramanujan\_machine\_discovers\_a\_new/](https://www.reddit.com/r/math/comments/16003f7/the_ramanujan_machine_discovers_a_new/)  
> 22. Lecture 36: Symbolic Computation with sympy, [https://homepages.math.uic.edu/\~jan/mcs320/mcs320notes/lec36.html](https://homepages.math.uic.edu/~jan/mcs320/mcs320notes/lec36.html)  
> 23. The two-level multipair PSLQ algorithm \- David H Bailey, [https://www.davidhbailey.com/dhbpapers/pslqm2-alg.pdf](https://www.davidhbailey.com/dhbpapers/pslqm2-alg.pdf)  
> 24. Integer relation algorithm \- Wikipedia, [https://en.wikipedia.org/wiki/Integer\_relation\_algorithm](https://en.wikipedia.org/wiki/Integer_relation_algorithm)  
> 25. Using Integer Relations Algorithms for finding Relationships among, [https://chamberland.math.grinnell.edu/papers/pslq.pdf](https://chamberland.math.grinnell.edu/papers/pslq.pdf)  
> 26. PSLQ Algorithm Provides Better Way to Find Integer Relations, [https://www2.lbl.gov/Science-Articles/Archive/pi-algorithm.html](https://www2.lbl.gov/Science-Articles/Archive/pi-algorithm.html)  
> 27. \[2601.08461\] A Rigorous Proof of a Ramanujan Machine Identity for, [https://arxiv.org/abs/2601.08461](https://arxiv.org/abs/2601.08461)