# **Dynamic Multi-Task Curriculum Scheduling and Exploration Allocation in Neuro-Symbolic Reinforcement Learning**

## **Structural Diagnostics: Advantage Collapse and Dilution Dynamics**

The empirical stagnation observed in Run 005, wherein the policy failed to graduate from Stage 1 after 24,000 optimization steps, stems from structural incompatibilities among Group Relative Policy Optimization (GRPO), uniform prompt sampling, and sparse binary program verification. In neuro-symbolic program synthesis, execution feedback is binary: a candidate program either satisfies the full suite of input-output assertions $R(P) \= 1$ or fails $R(P) \= 0$. When policy gradient methods discard parametric critic baselines to evaluate advantages purely across small rollout groups, the mathematical structure of the advantage estimator imposes severe constraints on parameter updates.

### **Mathematical Mechanics of Advantage Collapse**

In standard GRPO within the Reinforcement Learning from Verifiable Rewards (RLVR) paradigm, an actor network generates a group of $G$ independent completions $\\mathcal{O}\_i \= \\{y\_i^{(1)}, y\_i^{(2)}, \\dots, y\_i^{(G)}\\}$ for a given prompt $q\_i$. The scalar outcomes $\\mathcal{R}\_i \= \\{r\_i^{(1)}, r\_i^{(2)}, \\dots, r\_i^{(G)}\\}$ are evaluated deterministically by an external verifier. The empirical group mean $\\mu\_{\\mathcal{R}\_i}$ and standard deviation $\\sigma\_{\\mathcal{R}\_i}$ are computed as:

$$\\mu\_{\\mathcal{R}\_i} \= \\frac{1}{G} \\sum\_{j=1}^G r\_i^{(j)}, \\quad \\sigma\_{\\mathcal{R}\_i} \= \\sqrt{\\frac{1}{G} \\sum\_{j=1}^G (r\_i^{(j)} \- \\mu\_{\\mathcal{R}\_i})^2}$$  
The standardized advantage $\\hat{A}\_i^{(j)}$ assigned to each sequence $y\_i^{(j)}$ is defined by:

$$\\hat{A}\_i^{(j)} \= \\frac{r\_i^{(j)} \- \\mu\_{\\mathcal{R}\_i}}{\\sigma\_{\\mathcal{R}\_i} \+ \\epsilon}$$  
where $\\epsilon \> 0$ provides numerical stability. Under binary evaluation where $r\_i^{(j)} \\in \\{0, 1\\}$, let $k\_i \= \\sum\_{j=1}^G r\_i^{(j)}$ denote the total number of successful program completions in group $G$. The sample variance simplifies analytically to:

$$\\sigma\_{\\mathcal{R}\_i}^2 \= \\frac{k\_i(G \- k\_i)}{G^2}$$  
Whenever all $G$ trajectories fail ($k\_i \= 0$) or all $G$ trajectories succeed ($k\_i \= G$), the within-group reward variance vanishes: $\\sigma\_{\\mathcal{R}\_i} \= 0$. In both situations, the numerator collapses identically:

$$r\_i^{(j)} \- \\mu\_{\\mathcal{R}\_i} \= c \- c \= 0 \\implies \\hat{A}\_i^{(j)} \= \\frac{0}{0 \+ \\epsilon} \= 0 \\quad \\forall j \\in \\{1, \\dots, G\\}$$  
The policy gradient contribution contributed by prompt $q\_i$ to the GRPO objective is:

$$g\_i(\\theta) \= \\frac{1}{G} \\sum\_{j=1}^G \\sum\_{t=1}^{\\vert{}y\_i^{(j)}\\vert{}} \\min \\left( \\rho\_{i, t}^{(j)} \\hat{A}\_i^{(j)}, \\, \\text{clip}(\\rho\_{i, t}^{(j)}, 1-\\varepsilon, 1+\\varepsilon) \\hat{A}\_i^{(j)} \\right) \\nabla\_\\theta \\log \\pi\_\\theta(y\_{i, t}^{(j)} \\mid q\_i, y\_{i, \<t}^{(j)})$$  
Consequently, whenever $k\_i \\in \\{0, G\\}$, the policy gradient evaluates to $g\_i(\\theta) \= \\mathbf{0}$. This condition is defined as advantage collapse. The Advantage Collapse Rate (ACR) over a mini-batch of $B$ unique prompts quantifies the fraction of prompts yielding null gradients:

$$\\text{ACR} \= \\frac{1}{B} \\sum\_{i=1}^B \\mathbb{I}\\left\[ k\_i \\in \\{0, G\\} \\right\]$$  
When $k\_i \\in \\{1, \\dots, G-1\\}$, non-zero advantage signals emerge. For a successful rollout ($r=1$) and an unsuccessful rollout ($r=0$), the standardized advantages evaluate to:

$$\\hat{A}^+ \= \\frac{1 \- k/G}{\\frac{\\sqrt{k(G-k)}}{G}} \= \\sqrt{\\frac{G-k}{k}}, \\quad \\hat{A}^- \= \\frac{0 \- k/G}{\\frac{\\sqrt{k(G-k)}}{G}} \= \-\\sqrt{\\frac{k}{G-k}}$$  
The ratio of magnitude between positive and negative reinforcement is strictly asymmetric:

$$\\left\\vert{} \\frac{\\hat{A}^+}{\\hat{A}^-} \\right\\vert{} \= \\frac{G \- k}{k}$$  
When an exploratory program succeeds for the first time in a rollout group ($k=1$), $\\hat{A}^+ \= \\sqrt{G \- 1}$ while each failure receives $\\hat{A}^- \= \-1/\\sqrt{G \- 1}$. With group size $G=4$, a single discovery yields $\\hat{A}^+ \= \\sqrt{3} \\approx 1.732$ and $\\hat{A}^- \= \-1/\\sqrt{3} \\approx \-0.577$. As $G$ increases, the positive reinforcement for isolated breakthroughs scales as $\\mathcal{O}(\\sqrt{G})$, creating strong directional gradients that anchor the newly discovered program logic.

### **Non-Zero Gradient Probability and Batch Degradation**

Assuming a sequence benchmark $i$ exhibits an underlying policy pass probability $p\_i \= P(R(P)=1 \\mid q\_i, \\pi\_\\theta)$, the number of successes $k\_i$ follows a binomial distribution $\\text{Binomial}(G, p\_i)$. The probability that prompt $q\_i$ yields an active, non-zero gradient signal is:

$$P(\\text{signal} \\mid p\_i, G) \= 1 \- P(k\_i \= 0\) \- P(k\_i \= G) \= 1 \- (1 \- p\_i)^G \- p\_i^G$$  
In an uncurated training pool where tasks are sampled uniformly, tasks at the capability frontiers ($p\_i \\to 0$ or $p\_i \\to 1$) provide near-zero training signal. When a task is difficult for the current policy—such as a complex recurrence relation where $p\_i \= 0.01$—the probability of obtaining a training signal with group size $G=4$ is:

$$P(\\text{signal} \\mid 0.01, 4\) \= 1 \- (0.99)^4 \- (0.01)^4 \\approx 0.0394 \\quad (3.94\\%)$$  
In a mini-batch of $B=8$ prompts, the expected number of effective prompts providing non-zero gradients is $E\[B\_{\\text{eff}}\] \= B \\cdot P(\\text{signal}) \= 8 \\times 0.0394 \\approx 0.315$. The probability that an entire batch of 8 prompts suffers from complete advantage collapse is:

$$P(\\text{Batch Gradient} \= \\mathbf{0}) \= \\prod\_{i=1}^B (1 \- P(\\text{signal} \\mid p\_i, G)) \= (1 \- 0.0394)^8 \\approx 0.725 \\quad (72.5\\%)$$  
Under these conditions, nearly three out of every four optimization steps execute forward and backward passes that evaluate to a zero vector. The remaining updates are driven by noisy, isolated hits on easier problems, causing gradient updates to fluctuate erratically without driving systematic credit assignment on challenging sequences.

### **Task Dilution and Parameter Drift**

The empirical failure in Run 005 was exacerbated by task dilution across the 524 On-Line Encyclopedia of Integer Sequences (OEIS) benchmarks. With batch size $B=8$ and 400 optimizer steps per epoch, the total number of prompts sampled per epoch is $N\_{\\text{epoch}} \= 8 \\times 400 \= 3,200$. Uniform sampling across 524 sequences results in an average visitation frequency of:

$$\\lambda\_{\\text{visit}} \= \\frac{3,200}{524} \\approx 6.11 \\text{ visits per sequence per epoch}$$  
The expected interval between successive visitations to the same sequence benchmark evaluates to:

$$\\Delta\_{\\text{interval}} \= \\frac{524}{B} \= \\frac{524}{8} \= 65.5 \\text{ gradient steps}$$  
During the 65 intervening gradient steps, the network parameters $\\theta$ are continuously updated on unrelated sequences exhibiting distinct mathematical dynamics, including modular arithmetic, polynomial sequences, and combinatorial recurrences. Because symbolic program synthesis operates over discontinuous optimization landscapes where single-token modifications alter execution traces, parameter drift across 65 gradient steps erases fragile weight configurations discovered during earlier rollouts.

The policy enters a limit cycle: when a rare positive program is found for a difficult sequence, the gradient update pulls parameters toward that solution subspace; over the next 65 steps, orthogonal gradients from other tasks overwrite these updates; by the time the sequence is revisited, the model has reverted to its prior state. The model stabilizes on a diffuse heuristic that solves only trivial linear patterns, preventing the rolling competence $C(S\_1)$ from ever reaching the 0.85 graduation threshold.

| Sequence Pass Rate (p) | Group Size (G) | Prompt Batch (B) | P(signal∣task) | Expected Active Tasks (E\[Beff​\]) | P(Batch=0) | Positive Advantage (A^+ for k=1) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **0.01** | 4 | 8 | 0.0394 | 0.32 | 0.7250 | \+1.732 |
| **0.01** | 8 | 4 | 0.0773 | 0.31 | 0.7250 | \+2.646 |
| **0.01** | 16 | 2 | 0.1485 | 0.30 | 0.7250 | \+3.873 |
| **0.01** | 32 | 1 | 0.2750 | 0.28 | 0.7250 | \+5.568 |
| **0.05** | 4 | 8 | 0.1855 | 1.48 | 0.1937 | \+1.732 |
| **0.05** | 8 | 4 | 0.3366 | 1.35 | 0.1937 | \+2.646 |
| **0.05** | 16 | 2 | 0.5599 | 1.12 | 0.1937 | \+3.873 |
| **0.05** | 32 | 1 | 0.8063 | 0.81 | 0.1937 | \+5.568 |
| **0.20** | 4 | 8 | 0.5888 | 4.71 | 0.0008 | \+1.732 |
| **0.20** | 8 | 4 | 0.8322 | 3.33 | 0.0008 | \+2.646 |
| **0.20** | 16 | 2 | 0.9719 | 1.94 | 0.0008 | \+3.873 |
| **0.50** | 4 | 8 | 0.8750 | 7.00 | 0.0000 | \+1.732 |
| **0.80** | 4 | 8 | 0.5888 | 4.71 | 0.0008 | \+0.577 ($k=3$) |
| **0.95** | 4 | 8 | 0.1855 | 1.48 | 0.1937 | \+0.577 ($k=3$) |

The table above illustrates the quantitative interaction between underlying sequence solvability $p$, group size $G$, and batch size $B$ under a constant computational budget of $M \= B \\cdot G \= 32$ rollouts per step. At low capability ($p \\le 0.05$), small group sizes ($G=4$) induce high per-task collapse rates ($81.5\\%$ to $96.1\\%$), whereas allocating compute to deeper groups ($G=16, 32$) raises task-level non-zero signal probabilities to between $56.0\\%$ and $80.6\\%$ while scaling positive advantage signals up to $+5.568$.

## **Automated Curriculum Learning and Non-Stationary Bandit Scheduling**

Staged curriculum gating introduces artificial bottlenecks: by requiring the entire cohort of tasks within stage $S\_k$ to satisfy $C(S\_k) \\ge 0.85$, a handful of intractable sequences can stall training across the entire distribution. To eliminate this bottleneck, task allocation must be reformulated as an adaptive scheduling problem that prioritizes tasks located within the agent's Zone of Proximal Development (ZPD)—the frontier where learning progress is maximal.

### **Non-Stationary Multi-Armed Bandit Formulations**

The task scheduling problem across $K \= 524$ discrete OEIS sequences can be formalized as an adversarial non-stationary multi-armed bandit (MAB). Unlike stochastic bandits where reward distributions remain static, an RL student policy is continuously changing, rendering task returns non-stationary. The EXP3.S (Exponential-weight algorithm for Exploration and Exploitation with Switching) framework provides theoretical guarantees against tracking shifting optima.

Let the set of sequences be $\\mathcal{T} \= \\{1, \\dots, K\\}$. At each training step $t$, the curriculum scheduler maintains a probability vector $\\mathbf{p}\_t \= (p\_{1, t}, \\dots, p\_{K, t})$ over the simplex $\\Delta^K$. The distribution is updated via exponential weighting combined with uniform exploration and a mixing parameter that prevents arm weights from decaying to zero:

$$w\_{i, t+1} \= w\_{i, t} \\exp \\left( \\frac{\\gamma\_{\\text{exp3}} \\hat{r}\_{i, t}}{K} \\right) \+ \\frac{e \\cdot \\alpha\_{\\text{exp3}}}{K} \\sum\_{j=1}^K w\_{j, t}$$  
$$p\_{i, t+1} \= (1 \- \\gamma\_{\\text{exp3}}) \\frac{w\_{i, t+1}}{\\sum\_{j=1}^K w\_{j, t+1}} \+ \\frac{\\gamma\_{\\text{exp3}}}{K}$$  
where $\\gamma\_{\\text{exp3}} \\in (0, 1\]$ balances uniform exploration, $\\alpha\_{\\text{exp3}} \> 0$ governs the switching rate to adapt to non-stationary competence shifts, and $\\hat{r}\_{i, t}$ is the importance-weighted feedback signal received from sequence $i$.

### **Proxies for Learning Progress under Binary Rewards**

The central challenge in symbolic program synthesis is defining the bandit feedback signal $r\_{i, t}$. Standard curriculum methods rely on dense value function losses or continuous temporal-difference (TD) errors. In critic-free GRPO with binary outcomes $R(P) \\in \\{0, 1\\}$, continuous learning progress must be inferred from rollout statistics, empirical performance histories, or policy parameter dynamics.

Tracking the derivative of competence over a trailing sliding window of attempts $W\_i \= \\{r\_{i, 1}, r\_{i, 2}, \\dots, r\_{i, \\vert{}W\\vert{}}\\}$ isolates learning velocity in Teacher-Student Curriculum Learning (TSCL). Let the window be partitioned into an earlier half $W\_i^{\\text{early}}$ and a later half $W\_i^{\\text{late}}$. The directional learning progress is defined as:

$$\\Delta C\_i \= \\text{mean}(W\_i^{\\text{late}}) \- \\text{mean}(W\_i^{\\text{early}})$$  
To maintain stability against catastrophic forgetting while prioritizing frontier tasks, the curriculum reward incorporates an absolute slope formulation:

$$r\_{i}^{\\text{TSCL}} \= \\vert{}\\Delta C\_i\\vert{} \+ \\eta\_{\\text{forget}} \\max(0, \-\\Delta C\_i)$$  
The secondary term acts as an alarm: if a previously mastered sequence experiences performance decay ($\\Delta C\_i \< 0$), the arm receives immediate priority reinforcement to restore performance before weights drift further.

Under binary feedback, the immediate outcome variance $\\text{Var}(R \\mid q\_i) \= p\_i(1 \- p\_i)$ peaks precisely when the agent is at the threshold of mastery ($p\_i \\approx 0.5$). Tasks with $p\_i \\to 0$ (unsolvable with current parameters) and $p\_i \\to 1$ (already mastered) evaluate to zero variance. The bandit reward can be parameterized directly via binomial entropy:

$$\\mathcal{H}(p\_i) \= \-p\_i \\log\_2(p\_i \+ \\epsilon) \- (1 \- p\_i) \\log\_2(1 \- p\_i \+ \\epsilon)$$  
Sampling proportional to $\\mathcal{H}(p\_i)$ concentrates exploration on sequences where the model generates both successes and failures within group $G$, directly minimizing the Advantage Collapse Rate.

Adapted from Prioritized Level Replay (PLR), Maximum Monte Carlo (MaxMC) scores tasks by the maximum return ever achieved relative to the current baseline:

$$S\_{\\text{MaxMC}}(i) \= R\_{\\max}(i) \- \\bar{R}(i) \= 1 \- p\_i \\quad (\\text{for } R\_{\\max}(i) \= 1)$$  
For tasks where the model has succeeded at least once in history ($R\_{\\max}=1$), the regret represents unfinished learning: if the current pass rate is low ($p\_i \= 0.1$), $S\_{\\text{MaxMC}} \= 0.9$, signaling high replay priority. For completely unmastered tasks ($R\_{\\max}=0$), MaxMC assigns zero score, preventing the policy from stalling on currently intractable sequences.

Recent advances in Unsupervised Environment Design (UED) demonstrate that true realized learning progress is proportional to the parameter displacement induced on the policy network. By performing a forward-backward pass for candidate sequence $q\_i$, Parameter Change Environment Design (PACE) evaluates the Euclidean norm of the policy gradient:

$$S\_{\\text{PACE}}(i) \= \\Vert{}\\nabla\_\\theta \\mathcal{L}\_{\\text{GRPO}}(\\theta; q\_i)\\Vert{}\_2^2$$  
This metric requires no value baseline and directly measures how much the current parameterization would update if trained on task $i$. Tasks suffering from advantage collapse yield $S\_{\\text{PACE}} \= 0$, while tasks with high gradient magnitude receive prioritized allocation.

### **Regret-Based and Open-Ended Curriculum Frameworks**

Advanced curriculum frameworks derived from open-ended learning and UED formalize the relationship between task generation, replay, and agent capabilities. In PAIRED (Protagonist Antagonist Induced Regret for Environment Design), curriculum design is framed as a three-player game between an environment generator, a student agent, and an antagonist agent. The generator maximizes regret, defined as the difference in expected return between the antagonist and the student. This prevents the generator from producing impossible tasks where both agents fail (regret is 0\) or trivial tasks where both succeed (regret is 0).

Prioritized Level Replay (PLR) replaces explicit generative adversaries with selective replay over randomly sampled environments. PLR maintains a buffer of seen environments, computes learning potential via trajectory scores, and selectively revisits levels with high regret. Robust Prioritized Level Replay ($\\text{PLR}^\\perp$) trains the student policy exclusively on prioritized replay trajectories, using random rollouts solely for curation.

ACCEL (Adversarially Compounding Complexity by Editing Levels) unifies regret curation with evolutionary mutation. Rather than generating levels from scratch or relying purely on random sampling, ACCEL applies small mutations to existing high-regret levels. In symbolic program synthesis, this corresponds to taking known sequence specifications and mutating boundary conditions, polynomial degrees, or recurrence depth to explore the immediate frontier of solvability. Similarly, Dreaming in Code (DiCode) synthesizes executable environment code variations to construct curricula that enable agents to acquire complex behaviors in open-ended settings.

| Metric / Framework | Mathematical Formulation | Primary Driver for Program Synthesis | Computational Overhead | Robustness to Binary Reward Sparsity |
| :---- | :---- | :---- | :---- | :---- |
| **TSCL Score Velocity** | $\\Delta C\_i \= \\bar{R}\_i^{\\text{late}} \- \\bar{R}\_i^{\\text{early}}$ | Tracks rate of empirical pass rate change over window $W\_i$ \[cite: 11\] | Negligible ($\\mathcal{O}(1)$ moving average) | Medium; requires repeated hits to form a velocity |
| **Binomial Dispersion** | $p\_i(1 \- p\_i)$ | Targets intermediate tasks with $p\_i \\approx 0.5$ to prevent advantage collapse | Minimal (scalar counter tracking) | High; drops unsolvable ($p=0$) and solved ($p=1$) tasks |
| **MaxMC (PLR)** | $R\_{\\max}(i) \- \\bar{R}(i) \= 1 \- p\_i$ | Approximates regret for any sequence solved at least once | Minimal (stores binary max and mean) | High; anchors on historical discoveries |
| **PACE** | $\\Vert{}\\nabla\_\\theta \\mathcal{L}\_i\\Vert{}\_2^2$ | Direct measure of policy parameter change under task gradient | Low (gradient $\\ell\_2$ norm computed during backward pass) | Very High; zero for collapsed batches, large for learnable batches |
| **MNA Regret** | $\\frac{1}{T} \\sum\_t \\max(0, \-A\_t)$ | Measures generalized negative advantage along execution traces | Moderate (requires step-level advantage calculation) | High; eliminates baseline overestimation bias |
| **EXP3.S Bandit** | $p\_{i} \\propto w\_i \\exp(\\gamma \\hat{r}\_i / K) \+ \\alpha/K$ | Optimal theoretical bounds for non-stationary arm switching | Low (exponential weight maintenance over $K$ tasks) | Very High; guaranteed minimum exploration floor |

## **Group Size versus Batch Size Allocation under Sparse Rewards**

A central systems trade-off in RLVR and GRPO is allocating fixed training compute $M$ between task diversity (batch size of prompts $B$) and rollout exploration per task (group size $G$), where $M \= B \\cdot G$.

### **GRPO as Contrastive Learning with Control Variates**

Theoretical reformulations demonstrate that GRPO functions as an implicit contrastive learning mechanism rather than purely a value-baseline policy gradient estimator. In standard PPO, a parametric critic $V\_\\phi(s)$ provides a baseline to reduce the variance of the policy gradient estimator. GRPO eliminates the parameterization and GPU memory footprint of $V\_\\phi$ by computing an empirical baseline across the response group $\\mathcal{O}\_i$.

For binary outcomes, the gradient contribution from group $\\mathcal{O}\_i$ can be decomposed into pairwise contrastive differences:

$$g\_i(\\theta) \\propto \\sum\_{j: r^{(j)}=1} \\sum\_{l: r^{(l)}=0} \\left\[ \\nabla\_\\theta \\log \\pi\_\\theta(y^{(j)} \\mid q\_i) \- \\nabla\_\\theta \\log \\pi\_\\theta(y^{(l)} \\mid q\_i) \\right\]$$  
This reveals that GRPO reduces variance by leveraging the within-prompt correlation between positive and negative trajectories. The mean reward of the group serves as a Monte Carlo control variate. When $G=2$ (minimalist 2-GRPO), contrastive updates are generated if and only if one candidate succeeds and the other fails ($k=1, G-k=1$).

If the model is operating on tasks where the baseline success probability is moderate ($p \\in \[0.2, 0.8\]$), 2-GRPO matches the gradient stability of large group sizes when the total prompt throughput is increased proportionally ($B \\cdot 2 \= M$). In such regimes, allocating compute to larger $B$ exposes the policy to a broader diversity of tasks, maximizing overall gradient efficiency.

### **Hit Probability Scaling and Gradient Variance in Sparse Regimes**

The contrastive equivalence of 2-GRPO breaks down in sparse-reward settings where $p\_i \\ll 0.1$. For a hard sequence with $p\_i \= 0.01$, the probability of sampling at least one correct program across independent rollouts scales exponentially with group size:

$$P(\\text{Hit} \\ge 1 \\mid G) \= 1 \- (1 \- p\_i)^G$$  
For $G=4$, $P(\\text{Hit}) \= 1 \- (0.99)^4 \\approx 0.0394$. For $G=16$, $P(\\text{Hit}) \= 1 \- (0.99)^{16} \\approx 0.1485$. For $G=32$, $P(\\text{Hit}) \= 1 \- (0.99)^{32} \\approx 0.2750$. For $G=64$, $P(\\text{Hit}) \= 1 \- (0.99)^{64} \\approx 0.4744$.

Consider the total gradient variance of the batch estimator. The global policy gradient is:

$$\\hat{g}\_{\\text{batch}} \= \\frac{1}{B} \\sum\_{i=1}^B g\_i(\\theta)$$  
where $g\_i(\\theta) \= \\mathbf{0}$ with probability $1 \- P(\\text{signal} \\mid p\_i, G)$. The variance of $\\hat{g}\_{\\text{batch}}$ across mini-batches has two components:

1. Intra-group variance arising from Monte Carlo token sampling within active groups.  
2. Inter-task sample-selection variance induced by random Bernoulli switching between active gradient updates ($g\_i \\ne \\mathbf{0}$) and complete advantage collapse ($g\_i \= \\mathbf{0}$).

When $G$ is small on hard tasks, the Bernoulli switching variance dominates: the effective batch size $B\_{\\text{eff}} \\sim \\text{Binomial}(B, P(\\text{signal}))$ fluctuates near zero. Even though a configuration with $B=32, G=4$ evaluates 128 rollouts across 32 tasks, if $p=0.01$, the expected active tasks per batch is only $E\[B\_{\\text{eff}}\] \= 32 \\times 0.0394 \\approx 1.26$. In $27.5\\%$ of steps, the entire batch of 32 tasks generates zero gradient despite consuming 128 forward executions.

Conversely, allocating compute to deep rollouts on fewer tasks ($B=4, G=32$, also 128 total rollouts) ensures that for each sampled task, $P(\\text{signal}) \= 0.275$. The expected active tasks is $E\[B\_{\\text{eff}}\] \= 4 \\times 0.275 \= 1.10$, but when a hit occurs, the advantage assigned to the discovered solution evaluates to $\\hat{A}^+ \= \\sqrt{(32 \- 1)/1} \= \\sqrt{31} \\approx 5.568$. This provides a concentrated, high-magnitude gradient step that firmly shifts model weights toward the successful symbolic execution branch. Small groups ($G=4$) bound the maximum advantage at $\\hat{A}^+ \= \\sqrt{3} \\approx 1.732$, providing insufficient update magnitude to anchor rare discoveries against parameter drift.

### **Optimal Allocation Regimes: Deep versus Broad Sampling**

The optimal allocation of compute between $G$ and $B$ depends on the current task pass rate $p$. In the sparse exploration regime ($p \< 0.05$), deep sampling with $G \\ge 16$ and $B \\le 4$ is required to cross the reachability threshold $1 \- (1-p)^G \> 0.3$ and maximize positive advantage magnitude $\\hat{A}^+ \= \\mathcal{O}(\\sqrt{G})$. Task diversity is intentionally restricted to prevent advantage collapse across the batch.

In the proximal development regime ($0.05 \\le p \\le 0.40$), balanced allocation ($G=8, B=8$ or $G=8, B=16$) stabilizes training. Hit probabilities are sufficiently high that group variance is consistently non-zero. Increasing $B$ reduces inter-task gradient variance across mathematical sequence classes.

In the consolidation or fluency regime ($p \> 0.40$), broad sampling with $G=2, 4$ and $B \\ge 16$ is optimal. As tasks reach high competence, large $G$ wastes compute: if $p=0.8$, $G=16$ results in frequent all-success groups and small positive advantages ($\\hat{A}^+ \\approx 0.5$). 2-GRPO or small $G$ with large $B$ provides the fastest convergence and lowest policy drift.

### **Adaptive Dynamic Group Sizing (Ada-G)**

To eliminate manual tuning, compute can be allocated dynamically per task based on real-time competence estimates $\\hat{p}\_i$. Given total rollout budget $M$, the scheduler assigns variable group sizes $G\_i$ subject to $\\sum\_{i=1}^B G\_i \= M$:

$$G\_i \= \\text{clip} \\left( \\left\\lceil \\frac{\\ln(1 \- P\_{\\text{target}})}{\\ln(1 \- \\max(\\hat{p}\_i, p\_{\\text{floor}}))} \\right\\rceil, \\, G\_{\\min}, \\, G\_{\\max} \\right)$$  
Setting $P\_{\\text{target}} \= 0.50$ guarantees that the group size allocated to sequence $i$ provides at least a $50\\%$ probability of sampling at least one positive rollout given current estimated competence $\\hat{p}\_i$. If a sequence has $\\hat{p}\_i \= 0.02$, the allocator sets $G\_i \= \\lceil \\ln(0.5)/\\ln(0.98) \\rceil \= 35$, dedicating deep exploration to that prompt; if $\\hat{p}\_i \= 0.5$, $G\_i \= \\lceil \\ln(0.5)/\\ln(0.5) \\rceil \= 1 \\implies G\_{\\min} \= 4$, freeing compute for other tasks.

| Training Regime | Pass Rate (p) | Optimal Configuration | Advantage Magnitude (A^+) | Primary Optimization Target |
| :---- | :---- | :---- | :---- | :---- |
| **Sparse Exploration** | $p \< 0.05$ | Deep Sampling: $G=16, 32; B=2, 4$ | $+3.87$ to $+5.57$ | Maximize hit probability; cross reachability threshold |
| **Proximal Development** | $0.05 \\le p \\le 0.40$ | Balanced: $G=8; B=8, 16$ | $+1.73$ to $+2.65$ | Maintain steady gradient flow; balance task variance |
| **Fluency & Mastery** | $p \> 0.40$ | Broad Sampling: $G=2, 4; B \\ge 16$ | $+0.58$ to $+1.73$ | Maximize prompt throughput; eliminate contrastive waste |

## **Replay Buffers and Anti-Forgetting Mechanisms for Policy Gradients**

Standard on-policy policy gradient algorithms (PPO, GRPO) discard trajectory experiences immediately following the gradient update. In multi-task symbolic program synthesis across 524 disparate mathematical rules, this memoryless property causes catastrophic forgetting: parameter updates that adapt the network to quadratic sequences disrupt weights encoding modular or bitwise representations.

### **The Stability-Plasticity Dilemma in Symbolic Domains**

The stability-plasticity dilemma is amplified in symbolic generation. Unlike continuous control where policies degrade gracefully through minor kinematic perturbations, program synthesis operates over discrete formal grammars: a single token change leads to complete program failure. Continuous gradient updates on new task domains induce catastrophic representational drift in earlier sequence families.

Empirical evaluations in continual RL demonstrate that without memory replay, performance on previously mastered tasks decays exponentially with the number of intervening gradient steps on out-of-distribution tasks. Overcoming this requires architectural decoupling: maintaining high plasticity on new exploratory tasks while enforcing rigid behavioral stability on mastered sequences.

### **Elite Demonstration Buffer (EDB)**

The Elite Demonstration Buffer (EDB) maintains a non-parametric archive of canonical, verified programs discovered throughout training. EDB is structured as an associative map $\\mathcal{D}\_{\\text{elite}} \= \\{i: \\mathcal{B}\_i\\}\_{i=1}^K$, where each sequence bucket $\\mathcal{B}\_i$ stores a bounded set of elite solutions:

$$\\mathcal{B}\_i \= \\{ (P\_1, \\ell\_1, \\tau\_1), (P\_2, \\ell\_2, \\tau\_2), \\dots, (P\_E, \\ell\_E, \\tau\_E) \\}$$  
A program $P$ is eligible for entry into $\\mathcal{B}\_i$ if and only if it passes all verification test cases $R(P) \= 1$. Solutions within $\\mathcal{B}\_i$ are prioritized using Minimum Description Length (MDL) and execution efficiency:

$$\\text{Score}(P) \= \-\\alpha\_{\\text{len}} \\cdot \\text{TokenLength}(P) \- \\alpha\_{\\text{time}} \\cdot \\text{ExecutionCycles}(P)$$  
Favoring shorter programs acts as an inductive bias (Occam's razor), penalizing memorized constants or bloated unrolled logic in favor of compact recursive or closed-form expressions. To prevent buffer redundancy, programs are transformed into canonical Abstract Syntax Tree (AST) representations, deduplicating structurally isomorphic programs to maintain syntactic diversity in $\\mathcal{B}\_i$.

### **Co-Optimizing On-Policy GRPO with Prioritized Off-Policy Replay**

To combine on-policy RL exploration with offline stability, training batches are partitioned into on-policy exploration prompts and replay prompts drawn from the EDB. The unified multi-task objective combines the clipped surrogate GRPO objective on active exploratory tasks with an off-policy Supervised Fine-Tuning (SFT) consistency loss on archived elite programs, regularized by reference policy Kullback-Leibler (KL) divergence:

$$\\mathcal{L}\_{\\text{unified}}(\\theta) \= \\mathcal{L}\_{\\text{GRPO}}(\\theta; \\mathcal{D}\_{\\text{active}}) \+ \\lambda\_{\\text{replay}} \\mathcal{L}\_{\\text{SFT}}(\\theta; \\mathcal{D}\_{\\text{elite}}) \- \\beta\_{\\text{KL}} \\mathbb{D}\_{\\text{KL}}(\\pi\_\\theta \\parallel \\pi\_{\\text{ref}})$$  
The active GRPO loss ($\\mathcal{L}\_{\\text{GRPO}}$) is evaluated over on-policy rollouts sampled from curriculum-selected frontier tasks $\\mathcal{D}\_{\\text{active}}$, driving the discovery of new symbolic programs. The prioritized SFT replay loss ($\\mathcal{L}\_{\\text{SFT}}$) is evaluated over elite programs sampled from previously mastered sequences:

$$\\mathcal{L}\_{\\text{SFT}}(\\theta; \\mathcal{D}\_{\\text{elite}}) \= \\frac{1}{|\\mathcal{B}\_{\\text{replay}}|} \\sum\_{(q\_i, P^\*) \\in \\mathcal{B}\_{\\text{replay}}} \\frac{1}{|P^\*|} \\sum\_{t=1}^{|P^\*|} \\log \\pi\_\\theta(P\_t^\* \\mid q\_i, P\_{\<t}^\*)$$  
This term acts as an anchor in weight space, projecting gradient updates into subspaces that do not degrade likelihood on verified solutions. Reference-policy KL regularization penalizes excessive drift of the policy distribution $\\pi\_\\theta$ from a frozen reference model $\\pi\_{\\text{ref}}$, preventing representation collapse and retaining linguistic coherence.

### **Replay Sampling Strategies**

To maximize retention while minimizing interference with new task acquisition, replay sequences are drawn from $\\mathcal{D}\_{\\text{elite}}$ using two complementary non-uniform distributions. Vulnerability-weighted replay (CLEAR-style) samples sequences that have not been visited for many optimization steps with probability proportional to their elapsed dormancy $\\Delta t\_{\\text{dormant}} \= t\_{\\text{current}} \- t\_{\\text{last\\\_visit}}$. This directly eliminates the 65-step parameter drift observed in Run 005\. Interference-aware coreset replay selects replay prompts based on gradient alignment. If the on-policy task gradient is $g\_{\\text{active}}$, replay tasks are prioritized if their historical gradients have negative cosine similarity with $g\_{\\text{active}}$, directly resolving gradient conflicts before parameters are updated.

### **Connection to Iterated Learning Paradigms**

This hybrid architecture links directly to foundational frameworks in policy search and program learning. Expert Iteration (ExIt) decomposes reinforcement learning into separate exploration and generalization phases. In symbolic program synthesis, the policy generates candidate rollouts; verified solutions are collected into an expert dataset; the policy network is updated via supervised imitation on these self-discovered solutions. STaR (Self-Taught Reasoner) pairs rejection sampling with fine-tuning, filtering rollouts by terminal correctness and rationalizing incorrect steps post-hoc.

DreamCoder operates an iterative wake-sleep cycle where "wake" phases solve tasks using neural-guided search, and "sleep" phases compress discovered programs into reusable subroutines while training the neural recognition model on replayed solutions. The EDB serves the exact functional role of DreamCoder's program archive, preventing catastrophic forgetting across symbolic synthesis libraries. Furthermore, BREAD (Branched Rollouts with Expert Anchors) solves the cold-start problem on difficult tasks by using segments of successful historical traces as anchors, branching rollouts from intermediate states. This provides dense feedback on hard tasks that would otherwise yield persistent advantage collapse.

| Paradigm | Exploration Mechanism | Storage Representation | Update Target | Anti-Forgetting Property |
| :---- | :---- | :---- | :---- | :---- |
| **Vanilla GRPO** | On-policy sampling | Memoryless (discarded after step) | Policy parameters $\\theta$ | None; subject to rapid parameter drift |
| **CLEAR** | On-policy actor-critic | FIFO transition replay buffer | Actor-critic via V-trace | Replays past trajectories with off-policy correction |
| **Expert Iteration (ExIt)** | Tree search / rollouts | Verified positive traces dataset | Policy network (imitation) | Retrains periodically on historical solutions |
| **DreamCoder** | Enumerative search \+ sleep | Program AST library \+ sleep replays | Recognition model | Compresses solutions into shared symbolic primitives |
| **EDB \+ SFT Co-Optimization** | Dynamic Ada-G rollouts | Canonical MDL AST programs | Joint GRPO actor \+ SFT loss | Multi-task gradient anchor across all solved tasks |

## **The SYMPLE Unified Engine**

The proposed algorithmic framework integrates automated curriculum scheduling, dynamic group size allocation, and elite demonstration replay into a unified framework: SYMPLE (Symbolic Multi-Task Policy Learning Engine).

### **Operational Lifecycle**

The execution loop of SYMPLE progresses through six tightly coupled phases at each optimization step $t \\in \\{1, \\dots, T\_{\\max}\\}$:

Phase 1: Task Selection and Frontier Identification. The curriculum scheduler queries the EXP3.S bandit distribution over all $K=524$ tasks, where arm selection probabilities are modulated by empirical binomial dispersion $p\_i(1 \- p\_i)$ and score velocity $\\Delta C\_i$. Two frontier tasks $q\_1, q\_2$ are sampled from the active Zone of Proximal Development. Tasks with historical discoveries in the EDB whose pass rates have fallen receive immediate regret priority.

Phase 2: Dynamic Group Sizing (Ada-G). Given a rollout budget of $M\_{\\text{active}} \= 24$ across the active tasks, group sizes $G\_1, G\_2$ are computed dynamically via the Ada-G formula based on rolling pass rate estimates $\\hat{p}\_i$, subject to bounds $G\_{\\min}=8$ and $G\_{\\max}=16$. This concentrates sampling depth on harder frontier tasks to guarantee that the probability of at least one positive rollout exceeds $50\\%$.

Phase 3: Rollout Generation and Verification. Candidate programs are sampled autoregressively from the actor $\\pi\_\\theta$ for the selected tasks and evaluated in a sandboxed Python execution environment against ground-truth OEIS integer sequence test cases. Rolling success histories $W\_i$ and visitation timestamps $t\_{\\text{last}}\[i\]$ are updated immediately.

Phase 4: Buffer Ingestion and Advantage Recovery. Any rollout achieving full verification ($R=1$) is canonicalized via AST parsing, evaluated for token length, and ingested into the Elite Demonstration Buffer $\\mathcal{B}\_i$. If a selected task experiences complete failure ($k\_i \= 0$) but contains an existing solution in $\\mathcal{B}\_i$, virtual sample injection is triggered: a synthetic positive return is introduced into the group statistics to restore non-zero advantage normalization without additional environment rollouts.

Phase 5: Off-Policy Elite Replay Selection. Two mastered sequences with non-empty elite buffers are sampled according to dormancy priority, favoring tasks that have remained unvisited for the longest duration. Their shortest canonical programs are retrieved to serve as supervised consistency targets.

Phase 6: Multi-Objective Optimization and Bandit Update. The policy parameters $\\theta$ are updated by backpropagating the joint objective: on-policy GRPO clipped surrogate loss over the 24 active rollouts, prioritized SFT cross-entropy loss over the replayed elite programs, and KL divergence regularization against $\\pi\_{\\text{ref}}$. The bandit weights in EXP3.S are updated using importance-weighted learning progress feedback, shifting sampling mass toward sequences demonstrating rapid gains or acute forgetting.

| Hyperparameter / Component | Production Value | Engineering Specification & Operational Function |
| :---- | :---- | :---- |
| **Total Rollout Budget ($M$)** | 32 rollouts / step | Fixed GPU execution ceiling per training step |
| **Active Allocation ($M\_{\\text{active}}$)** | 24 rollouts / step | Dedicated to on-policy frontier exploration ($B\_{\\text{active}} \= 2$) |
| **Dynamic Rollout Bounds** | $G\_{\\min} \= 8, G\_{\\max} \= 16$ | Enforces deep sampling on frontier tasks ($p \< 0.1$) |
| **Replay Batch ($B\_{\\text{replay}}$)** | 2 sequences / step | Off-policy demonstration batch retrieved from EDB |
| **EDB Capacity per Sequence ($E$)** | 4 programs | Stores top-4 shortest canonical AST solutions per sequence |
| **Replay Loss Weight ($\\lambda\_{\\text{replay}}$)** | 0.50 | Scales SFT consistency gradient relative to GRPO gradient |
| **Reference KL Weight ($\\beta\_{\\text{KL}}$)** | 0.02 | Bounds policy drift relative to frozen reference model $\\pi\_{\\text{ref}}$ \[cite: 39, 40\] |
| **EXP3.S Exploration ($\\gamma\_{\\text{exp3}}$)** | 0.15 | Enforces non-zero sampling floor across all 524 benchmarks |
| **EXP3.S Switching ($\\alpha\_{\\text{exp3}}$)** | 0.05 | Controls adaptation rate to non-stationary competence shifts |
| **Competence Window ($\\vert{}W\\vert{}$)** | 20 attempts | Sliding window tracking rolling pass probability $\\hat{p}\_i$ \[cite: 11\] |
| **Target Hit Probability ($P\_{\\text{target}}$)** | 0.50 | Target solvability threshold for dynamic group sizing |

### **Comparison Across Architectural Paradigms**

Contrasting SYMPLE with baseline reinforcement learning and curriculum designs highlights how the framework overcomes the specific empirical failure modes of Run 005:

| System Dimension | Baseline GRPO (Run 005\) | Prioritized Level Replay (PLR) | Expert Iteration (ExIt) | SYMPLE Architecture |
| :---- | :---- | :---- | :---- | :---- |
| **Curriculum Strategy** | Staged gating ($C(S\_k) \\ge 0.85$) | Regret buffer sampling | Periodic offline retraining | EXP3.S Bandit targeting ZPD |
| **Task Allocation per Step** | Uniform within stage ($B=8, G=4$) | Mixture: random \+ replay | Uniform instruction batches | Dynamic Ada-G ($B=2, G=12$ exploratory) |
| **Collapse Mitigation** | None (Gradients vanish at $k \\in \\{0, G\\}$) | Value function critic | Supervised cross-entropy loss | Virtual Sample Injection \+ ZPD filtering |
| **Anti-Forgetting Mechanism** | None (Memoryless discard) | Environment level re-visitation | Retraining on curated static traces | Interleaved GRPO \+ EDB SFT Replay |
| **Memory Buffer Structure** | None | Environment seed registry | Filtered offline JSONL dataset | AST Canonicalized MDL Program Archive |
| **Per-Step Compute Cost** | 32 forward evaluations | 32 forward runs \+ critic updates | High-cost decoupled batch sampling | 24 forward rollouts \+ 2 SFT backward targets |

## **Synthesis and Strategic Recommendations**

The failure of Run 005 was not driven by insufficient neural representation capacity, but by systemic optimization pathologies: advantage collapse within shallow rollout groups ($G=4$) and extreme task dilution across 524 uncurated benchmarks that allowed continuous parameter drift to erase newly discovered symbolic solutions.

To ensure robust graduation across the full sequence spectrum, the training infrastructure should be reconfigured according to three primary interventions:

First, compute allocation must be shifted immediately from broad, shallow sampling ($B=8, G=4$) to concentrated frontier exploration ($B=2, G=12$). Deepening rollout groups increases the per-task probability of sampling at least one correct program on hard sequences from $3.9\\%$ to over $40\\%$, while increasing the positive advantage signal from $+1.73$ to $+3.31$. Virtual sample injection provides a numerical safety net, ensuring that batches containing unmastered sequences never evaluate to zero gradient.

Second, rigid stage gates should be replaced with continuous non-stationary bandit scheduling via EXP3.S guided by binomial dispersion $p\_i(1-p\_i)$ and score velocity. De-emphasizing sequences that are either already mastered ($p \\to 1$) or temporarily unreachable ($p \\to 0$) concentrates gradient updates on tasks within the Zone of Proximal Development, eliminating the dilution bottleneck that stalled Run 005\.

Third, an Elite Demonstration Buffer paired with vulnerability-weighted SFT replay must be integrated directly into the GRPO optimization step. Archiving verified programs in canonical AST form and replaying them with cross-entropy loss anchors model weights against parameter drift. This resolves the 65-step visitation gap, ensuring that capabilities acquired on linear and periodic sequences remain preserved as the policy acquires complex recurrence and combinatorial synthesis rules.

