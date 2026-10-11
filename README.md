<p align="center">
  <img src="assets/cover.png" alt="Post-Training Resources" width="100%">
</p>

# Post-Training Resources

A curated resource list for learning model post-training and building useful applications.

The reading path begins with training infrastructure and continual pretraining for domain knowledge, then moves through imitation learning, preference optimization, on-policy distillation, RL, agentic RL, and multimodal RL. Read **Start here** first, then follow the numbered sections. Within each topic, resources progress from introductory explanations and guided examples to specialized methods and larger systems. The order follows prerequisites and increasing difficulty.

The core list favors foundational work, peer-reviewed research, official documentation, and reproducible training recipes. Recent preprints are included only when they provide a useful method or open recipe.

What each method is for:

- **Continual pretraining (CPT):** Keep training on domain text so the model learns that field's language and knowledge.
- **Imitation learning:** Show the model examples of the answers or actions you want it to produce.
- **Preference optimization:** Show it which answer you prefer, so it learns to favor answers like that.
- **Reinforcement learning (RL):** Let the model try a task and reward it when it succeeds. That could mean solving a math problem, getting code to pass tests, or finishing a workflow in an app.

## Contents

- [Start here: the minimum mental model](#start-here-the-minimum-mental-model)
- [1. Training infrastructure](#1-training-infrastructure)
  - [Memory and compute fundamentals](#memory-and-compute-fundamentals)
  - [LoRA and parameter-efficient fine-tuning](#lora-and-parameter-efficient-fine-tuning)
  - [Distributed training and scaling](#distributed-training-and-scaling)
  - [Profiling training runs](#profiling-training-runs)
  - [Framework implementation companions](#framework-implementation-companions)
- [2. Continual pretraining and domain knowledge](#2-continual-pretraining-and-domain-knowledge)
  - [Domain adaptation and knowledge retention](#domain-adaptation-and-knowledge-retention)
- [3. Imitation learning](#3-imitation-learning)
  - [Supervised fine-tuning and demonstration data](#supervised-fine-tuning-and-demonstration-data)
  - [Knowledge Distillation](#knowledge-distillation)
- [4. Preference optimization](#4-preference-optimization)
  - [Preference feedback and reward modeling](#preference-feedback-and-reward-modeling)
  - [Preference objectives](#preference-objectives)
- [5. On-Policy Distillation](#5-on-policy-distillation)
- [6. RL](#6-rl)
  - [RLVR algorithms and training](#rlvr-algorithms-and-training)
  - [RL environments and harnesses](#rl-environments-and-harnesses)
  - [RL training infrastructure](#rl-training-infrastructure)
- [7. Agentic RL](#7-agentic-rl)
  - [Agent learning, tools, and trajectory data](#agent-learning-tools-and-trajectory-data)
  - [Search, research, and writing](#search-research-and-writing)
  - [Code and software engineering](#code-and-software-engineering)
  - [Computer use](#computer-use)
  - [Business workflows](#business-workflows)
- [8. Multimodal RL](#8-multimodal-rl)
  - [Vision-language models](#vision-language-models)
  - [Speech and audio](#speech-and-audio)
  - [Image and video generation](#image-and-video-generation)
- [Evaluation and alignment](#evaluation-and-alignment)
  - [Benchmarks and evaluation](#benchmarks-and-evaluation)
  - [Safety and instruction hierarchy](#safety-and-instruction-hierarchy)
- [Frontier](#frontier)
  - [Model reports and training accounts](#model-reports-and-training-accounts)
- [Source policy](#source-policy)

## Start here: the minimum mental model

Assumes basic Python, PyTorch, and an understanding of next-token prediction. Read these in order; begin with each paper's overview and method before its derivations.

1. [Memory and compute fundamentals](#memory-and-compute-fundamentals) - Estimate the cost of a small run.
2. [LoRA and parameter-efficient fine-tuning](#lora-and-parameter-efficient-fine-tuning) - Learn adapter basics before training.
3. [Continual pretraining and domain knowledge](#domain-adaptation-and-knowledge-retention) - See when domain corpus training fits.
4. [Supervised fine-tuning](#supervised-fine-tuning-and-demonstration-data) - Run a small example based on demonstrations.
5. [Evaluation fundamentals](#benchmarks-and-evaluation) - Set a held-out baseline and compare checkpoints.
6. [Preference optimization](#preference-objectives) - Learn chosen and rejected response training.
7. [On-policy distillation](#5-on-policy-distillation) - Study teacher feedback on student outputs.
8. [RLVR](#rlvr-algorithms-and-training) - Start with math reasoning and GRPO.
9. [Agentic RL](#7-agentic-rl) - Move to tools and multi-step tasks.

Complete one small SFT run and an evaluation before moving on. Continue through imitation learning, preference optimization, on-policy distillation, and RL before agentic or multimodal RL. Return to distributed training and rollout infrastructure when implementing larger experiments; model reports are advanced reference reading.

## 1. Training infrastructure

Start with memory, compute, and adapters. Return to distributed training as your experiments grow; RL-specific systems appear under [RL training infrastructure](#rl-training-infrastructure).

### Memory and compute fundamentals

Begin here. Estimate the cost of a small training run before choosing a framework.

- [How To Scale Your Model: rooflines](https://jax-ml.github.io/scaling-book/roofline/) - Understand compute, memory bandwidth, and communication limits.
- [How To Scale Your Model: transformer math](https://jax-ml.github.io/scaling-book/transformers/) - Estimate parameters, training FLOPs, and memory requirements.
- [Activation checkpointing techniques](https://pytorch.org/blog/activation-checkpointing-techniques/) - Trade recomputation for lower activation memory.

### LoRA and parameter-efficient fine-tuning

Read the conceptual guide before the papers. Use LoRA for an initial experiment, then study the extra memory savings and tradeoffs of QLoRA.

- [PEFT: LoRA concepts](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) - Frozen base weights, trainable low-rank adapters, rank, and target modules.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - The adapter parameterization and original experiments.
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) - Tests when LoRA can match full fine-tuning for SFT and RL, including rank, learning rate, batch size, and which layers to adapt.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - Train adapters with a quantized base model to reduce memory use.

### Distributed training and scaling

Return after completing a small training run. Learn gradient synchronization, then state sharding, parallelism, and expert communication.

- [Getting started with DistributedDataParallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) - Replicate a model across GPUs and synchronize gradients.
- [Getting started with FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) - A worked tutorial for sharding model states.
- [How To Scale Your Model: training](https://jax-ml.github.io/scaling-book/training/) - Compare data, tensor, pipeline, and expert parallelism.
- [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook) - A practical guide to combining parallelism, memory optimizations, and communication strategies for efficient training across GPU clusters.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) - Sharding optimizer states, gradients, and parameters to reduce per-device memory.

### Profiling training runs

Use profiling to diagnose an observed bottleneck in compute, memory, or communication.

- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) - Record operator time, memory use, and training traces.
- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/) - Inspect CPU/GPU timelines, synchronization, and communication.

### Framework implementation companions

Choose an implementation after understanding the relevant training method and system design. These links support the educational reading above.

- [TRL - Transformers Reinforcement Learning](https://huggingface.co/docs/trl/index) - Trainer documentation for SFT, preference optimization, and RL.
- [huggingface/trl](https://github.com/huggingface/trl) - Source code companion to the TRL trainer documentation.
- [LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models](https://arxiv.org/abs/2403.13372) - A unified fine-tuning interface covering datasets, adapters, and training methods.
- [Accelerate](https://huggingface.co/docs/accelerate) - Launch and coordinate PyTorch training across devices.
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Implementation of model parallelism for large language model training.

## 2. Continual pretraining and domain knowledge

Continual pretraining (CPT) continues the next-token objective on a domain corpus before instruction tuning. It can adapt a model to domain language and knowledge; evaluate factual access, retention, and downstream task performance.

### Domain adaptation and knowledge retention

Begin with the training loop and domain-adaptation paper. Then study how successive weight updates affect knowledge access and forgetting. CPT is an optional stage when the target domain warrants it.

- [Causal language modeling tutorial](https://huggingface.co/docs/transformers/v4.49.0/tasks/language_modeling) - Prepare text, tokenize and chunk a corpus, and train with next-token loss as a starting loop for continued pretraining.
- [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964) - Foundational experiments in domain-adaptive and task-adaptive pretraining.
- [Can a Language Model Learn Facts Continually in Its Weights?](https://labs.baseten.co/articles/can-a-language-model-learn-facts-continually-in-its-weights) - Tests whether repeated fact updates stick in model weights, and what happens to earlier facts after later updates.
- [Continual learning and the post monolith AI era](https://labs.baseten.co/articles/continual-learning) - A discussion of why continual updates are hard for general models, and why narrower specialists may handle them better.

## 3. Imitation learning

Learning from demonstrations and teacher outputs, from supervised fine-tuning to distillation.

### Supervised fine-tuning and demonstration data

Begin with SFT and instruction-data examples, then study synthetic demonstrations and distillation.

- [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
- [Iterative SFT (iSFT): Dense Reward Learning](https://labs.baseten.co/articles/iterative-sft) - Build training examples by repairing model attempts until they pass a task evaluator, then train on the accepted answers. Includes medical scribe experiments.
- [Post-Training Science for Supervised Fine-Tuning](https://labs.baseten.co/articles/post-training-science-for-supervised-fine-tuning) - Compare learning rate, batch size, LoRA, epochs, and data across model families and real tasks, with downstream evaluations.

### Knowledge Distillation

After SFT, learn how a student model can imitate a teacher. Continue to [On-Policy Distillation](#5-on-policy-distillation) for distillation on student-generated outputs.

- [A Survey on Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2402.13116)

## 4. Preference optimization

Learning from comparison data, reward models, and direct preference objectives.

### Preference feedback and reward modeling

Start with the RLHF pipeline and human comparisons. Continue to reward modeling, AI feedback, failure modes, and the statistical model behind pairwise preferences.

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - InstructGPT: demonstrations, preference reward modeling, and RLHF.
- [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
- [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)

### Preference objectives

Read the RLHF overview first. Begin with DPO and its simpler reference-free variant, then binary feedback, theoretical objectives, and token-level optimization.

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) - DPO: learn directly from preference pairs.
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) - SimPO: preference optimization without a reference model.
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) - KTO: alignment from desirable/undesirable feedback.
- [IPO: A General Theoretical Paradigm to Understand Learning from Human Preferences](https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html) - Identity Preference Optimization and a general preference-learning objective.

## 5. On-Policy Distillation

On-policy distillation (OPD) trains a student on feedback for its own sampled outputs. Begin with the practical overview, then study the core objective and an efficient implementation.

- [On-Policy Distillation - Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://proceedings.mlr.press/v235/ko24c.html)

## 6. RL

Reinforcement learning with verifiable rewards (RLVR) for tasks such as math reasoning and code generation.

### RLVR algorithms and training

Follow the policy-gradient foundation through RLHF algorithms, then compare the group-relative methods used for verifiable math and code rewards. Focus on each method's advantage estimate, clipping unit, and sampling strategy.

- [Proximal Policy Optimization (PPO)](https://spinningup.openai.com/en/latest/algorithms/ppo.html) - OpenAI's guide explains the clipped objective, advantage estimates, pseudocode, and implementation.
- [The N Implementation Details of RLHF with PPO](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo) - Reproduce the original OpenAI RLHF training setup and inspect the implementation choices behind PPO fine-tuning.
- [Putting RL back in RLHF: RLOO](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo) - CohereForAI researchers explain leave-one-out baselines, compare RLOO with PPO, and walk through the TRL trainer.
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - Introduces GRPO and applies it to verifiable mathematical reasoning.
- [GRPO in Open Instruct](https://allenai.github.io/open-instruct/algorithms/grpo/) - AI2's algorithm guide connects the GRPO objective to practical variants, configuration, and runnable training recipes.
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://dapo-sia.github.io/) - The authors' project page explains DAPO's sampling, clipping, token-level loss, and reward-shaping choices alongside code and data.
- [GSPO: Towards Scalable Reinforcement Learning for Language Models](https://qwenlm.github.io/blog/gspo/) - The Qwen team explains sequence-level importance ratios and why GSPO can stabilize RL training for MoE models.
- [MiniMax-M1 and CISPO](https://www.minimax.io/news/minimaxm1) - MiniMax's model report introduces CISPO's clipped importance weights and describes its use in long reasoning training.
- [GSM8K post-training worklog](https://heyyanshuman.com/posts/posttraining_gsm8k_worklog/) - A hands-on walkthrough of post-training a model for math reasoning.
- See the [AWS GRPO + TRL tool-calling walkthrough](#business-workflows) for a runnable applied recipe.

### RL environments and harnesses

Start with tasks, observations, actions, state, rewards, and termination conditions. Then study how harnesses collect trajectories, how training transfers across harnesses, and how execution environments scale.

- [The ultimate guide to RL environments: building and scaling them in the LLM era](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide) - Environment components, reward wiring, and worked comparisons of environment frameworks.
- [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments) - Discover and share community environments for RL training and downstream evaluation.
- [Prime Intellect verifiers](https://github.com/PrimeIntellect-ai/verifiers) - Build reusable environments and evaluations for training and testing language models.
- [The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) - Train inside existing agent harnesses with OpenEnv, Harbor, and TRL; study trajectory capture and transfer across harnesses.
- [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) - Environment and reward infrastructure for RL training.

### RL training infrastructure

Read after a small GRPO experiment. Start with generation fundamentals and the async framework survey, then study weight synchronization and individual system designs.

- [How To Scale Your Model: transformer inference](https://jax-ml.github.io/scaling-book/inference/) - KV caches, batching, and generation throughput underpin efficient RL rollouts.
- [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape) - Survey async RL architectures through rollout buffers, policy staleness, weight synchronization, partial rollouts, and distributed backends.
- [Async RL in Pure JAX](https://divyamakkar0.github.io/scaling-async-rl.html) - An engineering worklog on scaling an asynchronous RL stack, including inference, RDMA weight transfer, memory optimization, and sharding.
- [Shipping a Trillion Parameters With a Hub Bucket: Delta Weight Sync in TRL](https://huggingface.co/blog/delta-weight-sync) - A worked design for sending sparse weight changes through shared storage from the trainer to vLLM rollout servers.

## 7. Agentic RL

Reinforcement learning for tool use, multi-turn trajectories, and practical agent workflows.

### Agent learning, tools, and trajectory data

Start with agent-RL methods, then study search and software task recipes.

- [Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning](https://arxiv.org/abs/2511.14460)
- [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)

### Search, research, and writing

Study a small set of trained search agents and one open trajectory-and-SFT recipe. Keep the training method clear when comparing their results.

- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) - Search-R1: train interleaved reasoning and search.
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160)
- [Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses](https://arxiv.org/abs/2606.02373) - Train search agents with harness-managed working memory, evidence curation, verification, and budget-aware context.
- [OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis](https://arxiv.org/abs/2603.20278) - Open search-and-browse trajectory synthesis followed by supervised fine-tuning; this is an SFT recipe, not RL.


### Code and software engineering

Start after verifiable rewards and agent learning. Move from test-driven code generation to coding RL, repository-level environments, software agents, and self-play.

- [Training Software Engineering Agents and Verifiers with SWE-Gym](https://arxiv.org/abs/2412.21139)
- [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](https://arxiv.org/abs/2502.18449)
- [DeepSWE: Training a Fully Open-sourced, State-of-the-Art Coding Agent by Scaling RL](https://www.together.ai/blog/deepswe)
- [CWM: An Open-Weights LLM for Research on Code Generation with World Models](https://arxiv.org/abs/2510.02387)

### Computer use

Start after tool-use fundamentals. Learn the computer-use interface and visual grounding before native GUI models and multi-turn online RL.

- [UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2509.02544)

### Business workflows

A few applied cases for tool use and domain tasks. Pair each recipe with a held-out task and a cost measure.

- [FinQA agent training with rLLM](https://docs.rllm-project.com/cookbooks/finqa) - A financial question-answering workflow; evaluate calculation accuracy and evidence use.
- [Improving Multi-Turn Tool Use with Reinforcement Learning](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning) - A practitioner account of training multi-turn API workflows; evaluate task completion and recovery from tool failures.

## 8. Multimodal RL

Papers that train vision, audio, or visual-generation models with reinforcement learning, with the reward and policy update made explicit.

### Vision-language models

Start with verifiable rewards for visual tasks, then compare general visual RL with perception-only training.

- [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785) - ICCV 2025 paper applying GRPO with task-specific verifiable rewards, including IoU for object detection.
- [VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model](https://arxiv.org/abs/2504.07615) ([code](https://github.com/om-ai-lab/VLM-R1)) - Studies RL for visual understanding, including out-of-domain transfer, reward hacking, and training scale.
- [Stop Thinking, Start Looking: Efficient Post-Training for Multimodal Document Question Answering via Reasoning-Free Alignment](https://arxiv.org/abs/2607.14682) - ICML 2026 EMM-QA workshop paper comparing GRPO with and without reasoning traces for document question answering.

### Speech and audio

Study reasoning RL for audio understanding, then preference-based training for conversational quality.

- [Step-Audio 2 Technical Report](https://arxiv.org/abs/2507.16632) - A concrete audio RL recipe: two PPO stages move from a binary reasoning-length reward to learned preference scoring, followed by GRPO for audio perception.
- [Step-Audio-R1.5 Technical Report](https://arxiv.org/abs/2604.25719) - Shows how RLHF can address the limits of correctness-only audio rewards. It uses a rubric-conditioned reward model for explicit criteria and pairwise preferences for multi-turn conversational quality, optimized with a PPO-style objective.

### Image and video generation

Start with the policy-gradient formulation for diffusion, then compare KL-regularized and group-relative updates for image and video generation.

- [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301) - Introduces DDPO by treating denoising as a multi-step decision process and optimizing downstream rewards with policy gradients.
- [DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2305.16381) - NeurIPS 2023 paper combining online policy optimization with KL regularization for reward-model-guided image generation.
- [DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818) ([code](https://github.com/XueZeyue/DanceGRPO)) - Extends GRPO to diffusion and rectified-flow models for text-to-image, text-to-video, and image-to-video generation.

## Evaluation and alignment

Evaluate training on held-out tasks before comparing specialized benchmarks or safety studies.

### Benchmarks and evaluation

Begin with evaluation fundamentals:

- Define the target task and measurable success criteria.
- Keep evaluation examples separate from training data and reward tuning.
- Measure the starting model and compare checkpoints on the same held-out tasks.
- Inspect failures alongside aggregate scores; track cost and latency for application workflows.

Use a small set of established benchmarks that cover tool use, software, web, computer, and multi-turn tasks.

- [The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models](https://proceedings.mlr.press/v267/patil25a.html)
- [LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code](https://arxiv.org/abs/2403.07974)
- [GAIA: a benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983)
- [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
- [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://arxiv.org/abs/2404.07972)

### Safety and instruction hierarchy

Start with concrete behavior and instruction-priority examples. Then read the training objectives and progressively more involved attack, defense, and reward-hacking studies.

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents](https://arxiv.org/abs/2406.13352)

## Frontier

Original model reports and development accounts. Disclosure varies across sources; use the core sections for algorithm and implementation references.

### Model reports and training accounts

Optional advanced reading. These reports disclose training choices that complement the method and recipe sections above.

- [Tülu 3: The next era in open post-training - Ai2](https://allenai.org/blog/tulu-3-technical) - An open post-training pipeline covering supervised fine-tuning, preference data, and RL.
- [Rufus-Air: An Open LLM Post-Training Recipe](https://arxiv.org/abs/2609.29421) - A reproducible eight-stage pipeline covering SFT, reasoning and coding RL, agent training, search, and RLHF, with data, rewards, infrastructure, and stagewise results.
- [QwQ-32B: Embracing the Power of Reinforcement Learning](https://qwenlm.github.io/blog/qwq-32b/) - A first-party reasoning-model training account.
- [Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement](https://arxiv.org/abs/2409.12122) - Math-focused supervised fine-tuning and reinforcement learning.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) - A detailed large-model report with post-training and evaluation sections.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - Large-scale training and post-training details for an open model.
- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348) - Covers domain-specialist SFT and GRPO, a generative reward model, and on-policy distillation to combine specialists, alongside RL infrastructure for million-token rollouts and agent sandboxes.
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599) - A first-party report on scaling long-CoT RL.
- [Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning](https://arxiv.org/abs/2504.13914) - A reasoning model report with RL training details.
- [MiMo: Unlocking the Reasoning Potential of Language Model, From Pretraining to Posttraining](https://arxiv.org/abs/2505.07608) - A first-party account of a complete reasoning post-training pipeline.
- [MiMo-V2.6: Scaling Reinforcement Learning for Self-Improvement](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf) - A detailed account of scaling agentic RL across asynchronous training, multi-task environments and harnesses, and groupwise trajectory grading, with open training resources.
- [Phi-4-reasoning Technical Report](https://arxiv.org/abs/2504.21318) - Data and training details for reasoning models.
- [MiniMax M2.1: Post-Training Experience and Insights for Agent Models](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models) - First-party lessons from agent-model post-training.
- [Tongyi DeepResearch Technical Report](https://arxiv.org/abs/2510.24701) - An applied deep-research training and evaluation account.
- [LongCat-Flash-Thinking-2601 training report](https://tech.meituan.com/2026/02/02/LongCat-Flash-Thinking-2601-techreport.html) - First-party training report for a reasoning model.
- [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://arxiv.org/abs/2508.06471) - A model report spanning reasoning, coding, and agentic post-training.

## Source policy

Keep a resource only when it directly teaches post-training, supplies a reproducible recipe, or documents a training system. Prefer one authoritative source per idea. Add a paper plus code link only when the code is the actual implementation and adds practical value. Do not infer method or relevance from a title. Identify whether the source updates model weights, trains a reward, generates data, or only changes inference. Some recent papers are preprints; the cited venue is given when publication has been verified.

A core source must be one of the following:

- the paper that introduced the mechanism;
- the official documentation that defines it;
- the creator repository that implements it;
- a direct implementer report with code, measurements, or enough detail to understand the training setup.

Reported improvements need the model, data, reward, baseline, and evaluation method. Business use cases should distinguish measured results from suggested applications.

See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a resource.
