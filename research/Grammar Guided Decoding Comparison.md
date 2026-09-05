# **Architecture and Synthesis of Grammar-Guided Decoding for Autoregressive WebAssembly Code Generation**

## **Theoretical Foundations of Grammar-Guided Decoding**

Autoregressive language models synthesize structural code sequences by sequentially sampling tokens from a conditional probability distribution over a vocabulary $\\mathcal{V}$, conditioned on a prompt prefix and a latent context vector1. In domain-specific synthesis tasks such as the oeis-learn pipeline—where a Transformer Decoder maps a dense latent embedding representing an integer sequence to WebAssembly Text (WAT) programs—unconstrained autoregressive sampling frequently introduces syntax errors, unbalanced parenthetical trees, and invalid instruction sequences3. Grammar-Guided Decoding (GGD) resolves these failure modes by dynamically intersecting formal language constraints with the logit space of the decoder at every step of generation1.  
The logit modification process operates directly on the output logit vectors produced by the model prior to probability normalization1. At decoding step $t$, given the prefix token sequence $x\_{\<t} \= (x\_1, x\_2, \\dots, x\_{t-1})$, the model projects its final hidden state to an unnormalized logit vector $\\mathbf{s}\_t \\in \\mathbb{R}^{\\vert{}\\mathcal{V}\\vert{}}$1. Simultaneously, a symbolic parsing engine tracks the lexical and syntactic state of the generated prefix against a target grammar $\\mathcal{G}$2. The parser produces a binary validation mask $\\mathbf{m}\_t \\in \\{0, 1\\}^{\\vert{}\\mathcal{V}\\vert{}}$, where a bit entry $\\mathbf{m}\_{t, v} \= 1$ indicates that appending the string representation of token $v \\in \\mathcal{V}$ yields a syntactically valid prefix continuation under $\\mathcal{G}$, while $\\mathbf{m}\_{t, v} \= 0$ indicates a structural violation1.  
The engine enforces structural constraints by performing additive logit masking7:

$$\\tilde{\\mathbf{s}}\_{t, v} \= \\begin{cases} \\mathbf{s}\_{t, v} & \\text{if } \\mathbf{m}\_{t, v} \= 1 \\\\ \-\\infty & \\text{if } \\mathbf{m}\_{t, v} \= 0 \\end{cases}$$  
The conditional probability distribution $P(x\_t \\mid x\_{\<t}, \\mathcal{G})$ is then derived via the Softmax transformation over the masked logit vector $\\tilde{\\mathbf{s}}\_t$7:

$$P(x\_t \= v \\mid x\_{\<t}, \\mathcal{G}) \= \\frac{\\exp(\\tilde{\\mathbf{s}}\_{t, v})}{\\sum\_{w \\in \\mathcal{V}} \\exp(\\tilde{\\mathbf{s}}\_{t, w})}$$  
Tokens assigned $-\\infty$ receive zero probability mass, guaranteeing that any sampled execution trajectory strictly complies with the formal grammar rules7.  
Despite its structural guarantees, standard GGD introduces systemic challenges rooted in subword tokenization1. Subword tokenizers (such as Byte-Pair Encoding) chunk text based on statistical frequency across training corpora rather than formal lexical boundaries1. Consequently, individual tokens often span multiple formal grammar terminals, or conversely, single grammar terminals are split across several subword tokens1. This mismatch creates the Token-Grammar Misalignment Problem, wherein a parser evaluating partial subword tokens may falsely reject valid continuous byte sequences or permit tokens that create deterministic syntax dead-ends downstream1. Naive subword masking forces overly invasive constraints, distorting the model's natural output stream and causing artificial whitespace or indentation artifacts that degrade output quality1.  
Beyond lexical alignment, greedy logit masking induces Likelihood Misalignment and distribution distortion4. Truncating invalid tokens and renormalizing the remaining logit space alters the joint probability distribution of the model4. A token choice that appears locally valid at step $t$ can steer the decoder into a low-probability sub-region of the latent space, forcing the model to complete the program using unnatural code structures4.  
To mitigate probability distortion, two primary algorithmic frameworks have been introduced:

> * **Minimally Invasive Constrained Decoding (DOMINO):** Incorporates speculative decoding and subword-aligned pre-computation to allow the model to propose multi-token subword paths without immediate logit truncation1. DOMINO verifies whether the combined byte stream forms a valid grammatical continuation, intervening only when a structural violation occurs, preserving natural token distributions and accelerating generation throughput by up to $2\\times$ over unconstrained decoding1.  
> * **Grammar-Aligned Decoding (GAD) and Adaptive Sampling with Approximate Expected Futures (ASAp):** GAD formalizes constrained sampling to preserve the exact conditional probabilities of the model over the valid grammar subset4. The ASAp algorithm estimates the expected future probability mass of partial paths, adjusting local sampling probabilities based on the integrated likelihood of all valid future continuations rather than relying solely on single-step local validities4.

## **Framework Architecture and Comparative Analysis**

Modern constrained decoding engines implement distinct automaton models, compilation strategies, vocabulary matching techniques, and dynamic schema execution pipelines3.

### **Outlines: Precompiled Finite-State Machine Indexing**

Outlines converts regular expressions and context-free schemas into Deterministic Finite Automata (DFAs)3. Before inference begins, Outlines precomputes a static state-to-token transition matrix mapping every DFA state $q \\in Q$ to the exact set of valid vocabulary token IDs12. At runtime step $t$, logit mask generation requires only a single array lookup: valid\_tokens \= mask\_table\[current\_state\]12.  
While this design yields fast per-token execution ($\<10\\,\\mu\\text{s}$), it suffers from severe compilation bottlenecks12. Precomputing full state-to-token matrices for complex or recursive grammars can take seconds to minutes and consume significant CPU memory12. Furthermore, standard DFAs cannot natively parse arbitrary context-free recursion (such as nested S-expressions in WebAssembly) without manually capping recursion depth, limiting Outlines' utility for dynamic or highly nested code generation tasks2.

### **XGrammar and XGrammar-2: Pushdown Automata and Context Partitioning**

XGrammar introduces a byte-level Pushdown Automaton (PDA) framework tailored for context-free grammars in high-throughput inference engines like SGLang and vLLM2. To minimize runtime parsing overhead, XGrammar partitions the tokenizer vocabulary into context-independent tokens (structural syntax such as parentheses, brackets, and keywords whose validity depends strictly on local grammar states) and context-dependent tokens (variable identifiers and numerical literals requiring full stack evaluation)2. Context-independent masks are precomputed and stored in an adaptive token mask cache, while context-dependent tokens are evaluated dynamically against a persistent stack2. XGrammar further employs context expansion, leveraging lookahead assertions across single-reference rules to expand local contexts and convert context-dependent checks into pre-computable context-independent masks2.  
XGrammar-2 extends this architecture for dynamic workloads via an Earley-based parser backend, TagDispatch for structural dispatching, and a Cross-Grammar Cache7. The Cross-Grammar Cache builds finite-state machine reference graphs of input grammars and hashes their acyclic and cyclic sub-graphs, enabling instant reuse of cached mask data across distinct schemas that share underlying structural sub-rules7.

### **llguidance: Rust-Native Earley Parsing on Regex Derivatives**

Developed as the execution engine for Microsoft’s Guidance library, llguidance is a Rust engine optimized for dynamic schemas12. llguidance combines an Earley parser with regular expression derivatives, enabling simultaneous validation of recursive context-free structures and fine-grained field-level regex patterns12.  
Instead of precomputing state-to-token transition tables across the vocabulary, llguidance evaluates constraints on the fly by traversing a dynamic token trie12. The engine steps through byte representations of vocabulary tokens against the current Earley state, eliminating offline compilation delays ($0.05\\text{--}2\\,\\text{ms}$ cold setup time) while maintaining per-token masking latencies between $40\\text{--}60\\,\\mu\\text{s}$ and flat $p99$ tail latencies12.

### **Emerging Architectures: GRID**

GRID utilizes LALR(1) configuration tracking executed via an optimized Rust kernel (RustWalker)21. GRID achieves per-token mask calculation latencies of $3.6\\text{--}6.7\\,\\mu\\text{s}$ with zero false rejects21. It relies on single-flight artifact compilation and deferral scheduling guards to decouple CPU-bound parsing routines from GPU execution loops21.

| Architectural Attribute | Outlines | XGrammar (v1) | XGrammar-2 | llguidance | GRID |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Primary Automaton Model** | Deterministic Finite Automaton (DFA)12 | Byte-Level Pushdown Automaton (PDA)2 | Earley Parser \+ Structural FSM Graph7 | Earley Parser on Regex Derivatives12 | LALR(1) Configuration Tracker21 |
| **Supported Formal Grammar Class** | Regular Grammars / Bounded CFG3 | Context-Free Grammars (CFG)2 | Context-Free \+ Dynamic Tag Dispatching9 | Mixed CFG & Regex Derivatives12 | LALR(1) Context-Free Grammars21 |
| **Upfront Compilation Latency** | High ($100\\,\\text{ms}$ to minutes)12 | Moderate ($10\\text{--}100\\,\\text{ms}$)7 | Extremely Low ($\<1\\text{--}5\\,\\text{ms}$)9 | Virtual Zero ($0.05\\text{--}2\\,\\text{ms}$)12 | Low ($\<2\\,\\text{ms}$ warmup)21 |
| **Per-Token Masking Latency** | $\<10\\,\\mu\\text{s}$ (Matrix lookup)12 | $20\\text{--}35\\,\\mu\\text{s}$ \[cite: 2, 7\] | $\<15\\,\\mu\\text{s}$ \[cite: 9, 11\] | $40\\text{--}60\\,\\mu\\text{s}$ \[cite: 12\] | $3.6\\text{--}6.7\\,\\mu\\text{s}$ \[cite: 21\] |
| **Vocabulary Matching Engine** | Full DFA Matrix Lookup12 | Adaptive Mask Cache \+ Persistent Stack2 | Substructure Cache \+ Earley Trie Mask11 | Dynamic Tokenizer Trie Traversal12 | Trie Walk via RustWalker Kernel21 |
| **Dynamic Schema Adaptability** | Poor (Requires state re-indexing)12 | Moderate (Static schemas preferred)7 | High (Cross-Grammar Substructure Cache)9 | Native / Optimal (On-the-fly parsing)12 | High (Single-flight hash indexing)21 |
| **Memory Footprint** | High (Scales with state count $\\times \\vert{}\\mathcal{V}\\vert{}$)12 | Moderate (Partitioned mask cache)2 | Compact (Sub-graph deduplication)11 | Low (Dynamic trie \+ state stack)12 | Very Low (Configuration-sized state)21 |
| **Serving Framework Integration** | vLLM, SGLang, llama.cpp3 | vLLM, SGLang (Native Default)3 | vLLM, SGLang9 | vLLM, SGLang, llama.cpp12 | Custom vLLM LogitsProcessor21 |

## **WebAssembly Text S-Expression Formalization for oeis-learn**

In the oeis-learn architecture, WebAssembly Text (WAT) serves as an intermediate synthesis target for generated algorithms. WebAssembly programs are formatted as nested symbolic expressions (S-expressions), where modules, functions, parameters, local variables, and stack operations are represented as parenthesized prefix structures5.

### **Formal Grammar Representation**

To ensure strict syntactical generation, the target output space of the Transformer Decoder is bound to an Extended Backus-Naur Form (EBNF) context-free grammar covering essential WAT arithmetic, local variable handling, and structured control flow:

$$\\text{wat\\\_module} \\rightarrow \\text{"(" "module" } \\text{func\\\_def}^+ \\text{ ")"}$$

$$\\text{func\\\_def} \\rightarrow \\text{"(" "func" } \\text{identifier}^? \\text{ signature local\\\_decl}^\* \\text{ instruction}^+ \\text{ ")"}$$

$$\\text{signature} \\rightarrow \\text{param\\\_decl}^\* \\text{ result\\\_decl}^?$$

$$\\text{param\\\_decl} \\rightarrow \\text{"(" "param" } \\text{identifier}^? \\text{ valtype ")"}$$

$$\\text{result\\\_decl} \\rightarrow \\text{"(" "result" } \\text{valtype ")"}$$

$$\\text{local\\\_decl} \\rightarrow \\text{"(" "local" } \\text{identifier}^? \\text{ valtype ")"}$$

$$\\text{valtype} \\rightarrow \\text{"i32"} \\mid \\text{"i64"} \\mid \\text{"f32"} \\mid \\text{"f64"}$$

$$\\text{plain\\\_instr} \\rightarrow \\text{"local.get "} \\text{index} \\mid \\text{"local.set "} \\text{index} \\mid \\text{"local.tee "} \\text{index} \\mid \\text{"i32.const "} \\text{integer\\\_literal}$$

$$\\text{plain\\\_instr} \\rightarrow \\text{"i32.add"} \\mid \\text{"i32.sub"} \\mid \\text{"i32.mul"} \\mid \\text{"i32.div\\\_s"} \\mid \\text{"i32.rem\\\_s"} \\mid \\text{"i32.eqz"} \\mid \\text{"return"}$$

$$\\text{folded\\\_instr} \\rightarrow \\text{"(" plain\\\_instr instruction}^\* \\text{ ")"}$$

$$\\text{folded\\\_instr} \\rightarrow \\text{"(" "block" } \\text{identifier}^? \\text{ result\\\_decl}^? \\text{ instruction}^+ \\text{ ")"}$$

$$\\text{folded\\\_instr} \\rightarrow \\text{"(" "loop" } \\text{identifier}^? \\text{ result\\\_decl}^? \\text{ instruction}^+ \\text{ ")"}$$

$$\\text{index} \\rightarrow \\text{integer\\\_literal} \\mid \\text{identifier}$$

### **Syntactic versus Semantic Correctness**

Standard context-free GGD ensures structural correctness, such as balanced parentheses, valid opcode identifiers, and complete S-expressions3. However, standard context-free constraints cannot enforce context-sensitive semantic validity24. A program generated under context-free GGD may still fail sandbox execution due to semantic violations:

> * **Unbound Identifier References:** Emitting local.get $x when $x has not been declared in the function parameter or local declaration list5.  
> * **Stack Type Mismatches:** Invoking binary stack operations (e.g., i32.add) when the operand stack contains incompatible data types or insufficient values.  
> * **Invalid Branching Depth:** Executing a control jump (br 2\) when the surrounding control nesting depth is less than the target index.

### **Environment-Indexed Grammars and Refinement Orders**

Semantic correctness during decoding can be achieved by parameterizing the decoder with Environment-Indexed Grammars24. Under this approach, the active grammar $\\mathcal{G}\_\\Gamma$ dynamically updates alongside an environment state $\\Gamma\_t$ tracked during token generation24.  
The environment state is defined as $\\Gamma\_t \= (\\text{Vars}\_t, \\text{Types}\_t, \\text{Depth}\_t)$, where $\\text{Vars}\_t$ records declared variable identifiers, $\\text{Types}\_t$ tracks evaluation stack types, and $\\text{Depth}\_t$ stores active control block nesting levels24. When the decoder processes a variable declaration (e.g., (param $n i32)), the symbol $n is added to $\\text{Vars}\_t$24.  
When the parser encounters a rule requiring a variable index (e.g., local.get index), the logit mask generator queries $\\mathcal{G}\_{\\Gamma\_t}$24. The mask generator narrows the valid token set for index to include *only* identifiers present in $\\text{Vars}\_t$24.  
By defining a formal refinement order $\\mathcal{G}\_{\\Gamma'} \\sqsubseteq \\mathcal{G}\_\\Gamma$, the decoding engine enforces No-Ghost Soundness24. This theoretical guarantee ensures that the decoder cannot sample references to uninitialized variables or invalid branch targets, resolving both context-free syntax and static semantic constraints at inference time24.

## **Systems-Level Serving Economics and Execution Mechanics**

Deploying structured outputs in real-time continuous batching environments introduces trade-offs between Time-to-First-Token (TTFT), Inter-Token Latency (ITL), batch scaling, and CPU-GPU synchronization overhead9.

### **Latency Dynamics and Engine Trade-offs**

The total latency of a constrained code synthesis system combines initial schema compilation delays with per-token logit masking steps12. In static offline workloads, engines that rely on precomputed state matrices (such as Outlines) achieve minimal ITL ($\<10\\,\\mu\\text{s}$) because logit masking simplifies to direct memory indexing12. However, when confronted with dynamic or un-cached schemas, precomputed engines incur substantial TTFT delays as state tables are constructed at runtime12.  
Dynamic engines (such as llguidance) eliminate precomputation bottlenecks by parsing continuous byte streams on the fly, yielding immediate warm TTFTs ($0.05\\text{--}2\\,\\text{ms}$)12. Although dynamic trie matching introduces slightly higher $p50$ per-token overhead ($40\\text{--}60\\,\\mu\\text{s}$), it avoids TTFT compilation spikes and maintains stable $p99$ tail latencies across varied structural inputs12. Advanced engines (such as XGrammar-2) balance these dynamics by combining dynamic Earley parsing with Cross-Grammar Substructure Caching9. By caching shared sub-graphs across distinct grammars, XGrammar-2 achieves low TTFTs ($\\approx 0.7\\,\\text{ms}$) alongside minimal ITL overhead9.

### **CPU-GPU Synchronization and Continuous Batching**

In continuous batching inference servers (such as vLLM or SGLang running on high-throughput GPUs), host-device synchronization presents a primary operational bottleneck16. Calculating logit masks on the CPU while execution kernels run on the GPU requires pipeline synchronization2. If CPU mask generation stalls, GPU worker queues block, leading to hardware underutilization16.  
Modern serving architectures address host-device synchronization using three principal strategies:

> * **Asynchronous CPU Worker Pools:** Mask construction for step $t+1$ is dispatched asynchronously to dedicated CPU thread pools during the GPU forward pass of step $t$2.  
> * **Skip-a-Round Deferral Guards:** System architectures like GRID implement deferral scheduling guards21. If a complex parse walk misses its scheduling window, the engine defers that specific request from the current sampling round21. Unaffected requests in the batch proceed without interruption, preventing isolated cold parsing operations from blocking the entire batch21.  
> * **Vectorized GPU Bitmask Application:** Precomputed or dynamically constructed token bitmasks are transferred and applied using vectorized CUDA kernels directly on GPU memory, avoiding host-to-device transfer overhead2.

| Engine Architecture | Outlines | XGrammar (v1) | XGrammar-2 | llguidance | GRID |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Cold Schema Compilation Latency** | $1,000\\text{--}60,000\\,\\text{ms}$ \[cite: 12\] | $10\\text{--}100\\,\\text{ms}$ \[cite: 7, 11\] | $0.7\\text{--}5\\,\\text{ms}$ \[cite: 9, 11, 21\] | $0.05\\text{--}2\\,\\text{ms}$ \[cite: 12\] | $0.5\\text{--}2\\,\\text{ms}$ \[cite: 21\] |
| **Warm Schema TTFT Overhead** | $\<0.1\\,\\text{ms}$ \[cite: 12\] | $\<0.5\\,\\text{ms}$ \[cite: 2, 11\] | $\<0.1\\,\\text{ms}$ \[cite: 9, 11\] | $\<0.1\\,\\text{ms}$ \[cite: 12\] | $\<0.01\\,\\text{ms}$ \[cite: 21\] |
| **Inter-Token Latency ($p50$)** | $5\\text{--}10\\,\\mu\\text{s}$ \[cite: 12\] | $20\\text{--}35\\,\\mu\\text{s}$ \[cite: 2\] | $8\\text{--}15\\,\\mu\\text{s}$ \[cite: 9, 11\] | $40\\text{--}60\\,\\mu\\text{s}$ \[cite: 12\] | $3.6\\text{--}6.7\\,\\mu\\text{s}$ \[cite: 21\] |
| **Inter-Token Latency ($p99$)** | $\>500\\,\\mu\\text{s}$ \[cite: 12, 21\] | $\>200\\,\\mu\\text{s}$ \[cite: 21\] | $\<50\\,\\mu\\text{s}$ \[cite: 9, 11\] | $\<100\\,\\mu\\text{s}$ (Flattest)21 | $\<30\\,\\mu\\text{s}$ \[cite: 21\] |
| **Batch Scaling Efficiency** | Degrades at high concurrency16 | High (Partitioned masks)2 | Very High (Substructure reuse)9 | High (Low memory footprint)12 | Optimal (Deferral guards)21 |
| **Host-Device Sync Dependency** | High sync overhead16 | Overlapped GPU execution2 | Async pipeline dispatch9 | Overlapped CPU trie walk12 | Non-blocking RustWalker21 |

## **Strategic Synthesis and Architectural Selection for oeis-learn**

Synthesizing valid WebAssembly Text from dense latent vectors in the oeis-learn pipeline requires balancing formal syntactic guarantees, static semantic validation, low serving latency, and high batching throughput. Based on these constraints, the following architectural design choices are indicated for the generation engine:

### **Engine Selection**

The pipeline should adopt llguidance or XGrammar-2 as its primary constrained decoding engine. Outlines should be excluded from consideration due to state-space expansion overhead when compiling recursive S-expression grammars, which introduces high compilation latency2. llguidance provides zero compilation delays and stable $p99$ tail latencies, supporting dynamically structured WAT queries without TTFT penalties12. Alternatively, XGrammar-2 provides high throughput by leveraging its Cross-Grammar Substructure Cache to index static WAT productions (such as type keywords, structural brackets, and opcode identifiers) across batch requests9.

### **Dynamic Scope Validation**

The decoder should integrate Environment-Indexed Grammar extensions to enforce dynamic scope validation alongside context-free parsing24. By tracking declared function parameters and local variables within an environment state $\\Gamma\_t$, the engine dynamically restricts variable index tokens to bound symbols5. This ensures No-Ghost Soundness, preventing the model from outputting references to undeclared local variables or invalid branch targets24.

### **Token Alignment and Sampling Probability Preservation**

To protect generation quality from probability distortion, the runtime should incorporate subword-aligned validation mechanisms (such as DOMINO-style speculative parsing or llguidance dynamic trie matching)1. Validating proposed multi-token byte sequences against the WAT parser prior to logit modification prevents subword token alignment artifacts and preserves the model's natural output probabilities1.

### **Serving Integration**

The generation engine should be integrated directly into a continuous batching inference server (such as vLLM or SGLang) using asynchronous CPU mask calculation pools2. Implementing skip-a-round deferral guards ensures that host-side parsing checks run concurrently with GPU forward passes, eliminating pipeline stalls and enabling efficient, zero-defect WAT code generation21.

#### **Works cited**

> 1. Guiding LLMs The Right Way: Fast, Non-Invasive Constrained, [https://arxiv.org/html/2403.06988v1](https://arxiv.org/html/2403.06988v1)  
> 2. XGrammar: Flexible and Efficient Structured Generation ... \- arXiv, [https://arxiv.org/pdf/2411.15100?](https://arxiv.org/pdf/2411.15100)  
> 3. Uncovering Control-Plane Vulnerabilities in LLMs with Structured, [https://arxiv.org/html/2503.24191](https://arxiv.org/html/2503.24191)  
> 4. Grammar-Aligned Decoding \- arXiv, [https://arxiv.org/html/2405.21047v1](https://arxiv.org/html/2405.21047v1)  
> 5. Understanding WebAssembly text format \- MDN Web Docs, [https://developer.mozilla.org/en-US/docs/WebAssembly/Guides/Understanding\_the\_text\_format](https://developer.mozilla.org/en-US/docs/WebAssembly/Guides/Understanding_the_text_format)  
> 6. Grammar-Constrained Decoding for Structured NLP Tasks without, [https://arxiv.org/html/2305.13971v6](https://arxiv.org/html/2305.13971v6)  
> 7. 1 Introduction \- arXiv, [https://arxiv.org/html/2601.04426v1](https://arxiv.org/html/2601.04426v1)  
> 8. Flexible and Efficient Grammar-Constrained Decoding \- arXiv, [https://arxiv.org/pdf/2502.05111?](https://arxiv.org/pdf/2502.05111)  
> 9. XGrammar-2: Dynamic and Efficient Structured Generation Engine, [https://arxiv.org/pdf/2601.04426](https://arxiv.org/pdf/2601.04426)  
> 10. Lost in Space: Optimizing Tokens for Grammar-Constrained Decoding, [https://arxiv.org/html/2502.14969v1](https://arxiv.org/html/2502.14969v1)  
> 11. Efficient Dynamic Structured Generation Engine for Agentic LLMs, [https://arxiv.org/html/2601.04426v3](https://arxiv.org/html/2601.04426v3)  
> 12. Constrained Decoding and Structured Output Engines in Production, [https://www.llms.blog/posts/constrained-decoding-and-structured-output-engines-in-production-comparing-xgrammar-llguidance-outlines-and-llama-cpp-gbnf-architecture-fsm-compilation-token-masking-latency-and-serving](https://www.llms.blog/posts/constrained-decoding-and-structured-output-engines-in-production-comparing-xgrammar-llguidance-outlines-and-llama-cpp-gbnf-architecture-fsm-compilation-token-masking-latency-and-serving)  
> 13. A Unified Decoding Framework for Large Language Models \- arXiv, [https://arxiv.org/html/2601.07525v2](https://arxiv.org/html/2601.07525v2)  
> 14. Generating Structured Outputs from Language Models: Benchmark, [https://www.researchgate.net/publication/388231978\_Generating\_Structured\_Outputs\_from\_Language\_Models\_Benchmark\_and\_Studies](https://www.researchgate.net/publication/388231978_Generating_Structured_Outputs_from_Language_Models_Benchmark_and_Studies)  
> 15. Guided Decoding and Its Critical Role in Retrieval-Augmented, [https://huggingface.co/blog/nmmursit/guided-decoding](https://huggingface.co/blog/nmmursit/guided-decoding)  
> 16. General questions on structured output backend \- vLLM Forums, [https://discuss.vllm.ai/t/general-questions-on-structured-output-backend/1444](https://discuss.vllm.ai/t/general-questions-on-structured-output-backend/1444)  
> 17. 1 Introduction \- arXiv, [https://arxiv.org/html/2411.15100v3](https://arxiv.org/html/2411.15100v3)  
> 18. Guided Decoding and Its Critical Role in Retrieval-Augmented, [https://arxiv.org/html/2509.06631v1](https://arxiv.org/html/2509.06631v1)  
> 19. X Grammar | PDF | Automata Theory | Computing \- Scribd, [https://www.scribd.com/document/1004694945/x-Grammar](https://www.scribd.com/document/1004694945/x-Grammar)  
> 20. Bidirectional Translation Through Formal IR \- codelift, [https://codelift.space/research](https://codelift.space/research)  
> 21. GRID: Grammar-Railed Decoding for Enterprise SQL Generation, [https://arxiv.org/pdf/2607.11951](https://arxiv.org/pdf/2607.11951)  
> 22. guidance-ai/llguidance: Super-fast Structured Outputs \- GitHub, [https://github.com/guidance-ai/llguidance](https://github.com/guidance-ai/llguidance)  
> 23. S-expression \- Wikipedia, [https://en.wikipedia.org/wiki/S-expression](https://en.wikipedia.org/wiki/S-expression)  
> 24. Constrained LLM Generation over a Refinement Order of Grammar, [https://arxiv.org/abs/2607.18357](https://arxiv.org/abs/2607.18357)