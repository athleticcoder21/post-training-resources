<p align="center">
  <img src="assets/cover.png" alt="Post-Training Resources" width="100%">
</p>

# Post-Training Resources

A curated resource list for learning model post-training and building useful applications.

The list is ordered from supervised fine-tuning and preference learning to policy optimization, reasoning, training systems, agents, and multimodal applications. Read **Start here** first. After that, use it as a reference.

The core list uses original papers, official documentation, creator repositories, and direct implementation work. It covers both the underlying algorithms and recipes for business workflows.

## Contents

- [Start here: the minimum mental model](#start-here-the-minimum-mental-model)
- [1. Post-training foundations](#1-post-training-foundations)
  - [SFT, preference data, and RLHF](#sft-preference-data-and-rlhf)
  - [Preference optimization](#preference-optimization)
- [2. Policy optimization and reasoning](#2-policy-optimization-and-reasoning)
  - [Policy optimization and verifiable rewards](#policy-optimization-and-verifiable-rewards)
  - [Distillation](#distillation)
  - [Reasoning, process rewards, and search](#reasoning-process-rewards-and-search)
- [3. Training systems](#3-training-systems)
  - [Training frameworks and infrastructure](#training-frameworks-and-infrastructure)
- [4. Agents and applications](#4-agents-and-applications)
  - [Agent learning, tools, and trajectory data](#agent-learning-tools-and-trajectory-data)
  - [Search, research, and writing](#search-research-and-writing)
  - [Code and software engineering](#code-and-software-engineering)
  - [Computer use](#computer-use)
  - [Business workflows](#business-workflows)
- [5. Multimodal post-training](#5-multimodal-post-training)
  - [Vision-language models](#vision-language-models)
  - [Speech and audio](#speech-and-audio)
  - [Image and video generation](#image-and-video-generation)
- [6. Evaluation and alignment](#6-evaluation-and-alignment)
  - [Benchmarks and evaluation](#benchmarks-and-evaluation)
  - [Safety and instruction hierarchy](#safety-and-instruction-hierarchy)
- [Frontier](#frontier)
  - [Model reports and training accounts](#model-reports-and-training-accounts)
- [Source policy](#source-policy)

## Start here: the minimum mental model

Read these in order if you are new to the field.

1. [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) - Supervised fine-tuning with instruction and demonstration data.
2. [InstructGPT](https://arxiv.org/abs/2203.02155) - Demonstrations, preference reward modeling, and reinforcement learning from human feedback.
3. [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) - The clipped policy objective used in many RL training pipelines.
4. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - Learn a policy directly from chosen and rejected responses.
5. [DeepSeekMath](https://arxiv.org/abs/2402.03300) - Group Relative Policy Optimization and mathematical reasoning.
6. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - Reasoning with verifiable rewards, cold-start data, and distillation.
7. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - Process supervision and intermediate verification.
8. [HybridFlow](https://arxiv.org/abs/2409.19256) - The distributed training architecture behind veRL.

For a practical companion, use the [AReaL search agent](https://github.com/areal-project/AReaL/tree/main/examples/search_agent).

## 1. Post-training foundations

### SFT, preference data, and RLHF

- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - InstructGPT: demonstrations, preference reward modeling, and RLHF.
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
- [WizardLM: Empowering large pre-trained language models to follow complex instructions](https://arxiv.org/abs/2304.12244)
- [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons - JSTOR](https://www.jstor.org/stable/2334029)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)
- [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861)
- [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)

### Preference optimization

- [Token-level Direct Preference Optimization](https://arxiv.org/abs/2404.11999)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) - DPO: learn directly from preference pairs.
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) - KTO: alignment from desirable/undesirable feedback.
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) - SimPO: preference optimization without a reference model.
- [IPO: A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) - Identity Preference Optimization and a general preference-learning objective.

## 2. Policy optimization and reasoning

### Policy optimization and verifiable rewards

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - DeepSeekMath: the GRPO objective and math reasoning training.
- [ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models](https://arxiv.org/abs/2310.10505)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - PPO: clipped policy optimization.
- [hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)
- [SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild](https://arxiv.org/abs/2503.18892)
- [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) - Dr. GRPO and analysis of R1-Zero-style optimization.
- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)
- [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) - GSPO: sequence-level importance ratios and policy optimization.
- [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585) - MiniMax-M1: CISPO and efficient reasoning training.
- [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)
- [Reinforcement Pre-Training](https://arxiv.org/abs/2506.08007)
- [Fine-tune Small Language Models for Production-Grade Tool Calling with GRPO using Hugging Face TRL on Amazon SageMaker AI](https://builder.aws.com/content/35x6VR6kZYSn3JgNQmcNmIVK32Y/fine-tune-small-language-models-for-production-grade-tool-calling-with-grpo-using-hugging-face-trl-on-amazon-sagemaker-ai)
- [PRIME-RL/Eurus-2-RL-Data · Datasets at Hugging Face](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://seed.bytedance.com/en/public_papers/dapo-an-open-source-llm-reinforcement-learning-system-at-scale)
- [BytedTsinghua-SIA/DAPO](https://github.com/BytedTsinghua-SIA/DAPO)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) - DAPO: dynamic sampling, clipping, reward shaping, and a released training system.

### Distillation

- [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)
- [On-Policy Distillation - Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [A Survey on Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2402.13116)
- [Survey on Knowledge Distillation for Large Language Models: Methods, Evaluation, and Application](https://arxiv.org/abs/2407.01885)
- [Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation](https://arxiv.org/abs/2604.13010)
- [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- [Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016)
- [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://proceedings.mlr.press/v235/ko24c.html)
- [OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM)
- [MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training](https://mimo.xiaomi.com/papers/mopd.pdf)
- [Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2608.19098)
- [Open-MOPD: Multi-Teacher On-Policy Distillation Recipe](https://bytedtsinghua-sia.github.io/Open-MOPD/)
- [Black-Box On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2511.10643)
- [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734)
- [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626)

### Reasoning, process rewards, and search

- [Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- [Introducing OpenAI o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)
- [Competitive Programming with Large Reasoning Models](https://arxiv.org/abs/2502.06807)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) - DeepSeek-R1: reasoning RL, cold-start data, and distilled models.
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) - Allocate test-time computation between sampling and refinement.
- [Gemini 2.5: Deep Think is now rolling out](https://blog.google/products-and-platforms/products/gemini/gemini-2-5-deep-think/)
- [Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)
- [Extended thinking - Claude Platform Docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - Process supervision and step-level verification.
- [Process Reward Models That Think](https://arxiv.org/abs/2504.16828)
- [AI achieves silver-medal standard solving International Mathematical Olympiad problems - Google DeepMind](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)
- [DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition](https://arxiv.org/abs/2504.21801)
- [openai/prm800k](https://github.com/openai/prm800k)
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)
- [leanprover-community/mathlib4](https://github.com/leanprover-community/mathlib4)
- [Autoformalization with Large Language Models](https://arxiv.org/abs/2205.12615)
- [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240) - Generative verification through next-token prediction.
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)

## 3. Training systems

### Training frameworks and infrastructure

- [huggingface/trl](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_config.py)
- [huggingface/trl](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py)
- [Firecracker](https://firecracker-microvm.github.io/)
- [The Container Security Platform - gVisor](https://gvisor.dev/)
- [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298) - AReaL: asynchronous rollouts and large-scale RL training.
- [AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework](https://arxiv.org/abs/2510.04206)
- [THUDM/slime](https://github.com/THUDM/slime)
- [alibaba/ROLL](https://github.com/alibaba/ROLL)
- [veRL code sandbox setup](https://www.volcengine.com/docs/6460/1756203)
- [LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models](https://arxiv.org/abs/2403.13372)
- [TRL - Transformers Reinforcement Learning](https://huggingface.co/docs/trl/index)
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256) - HybridFlow: distributed RLHF architecture underlying veRL.
- [LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework for Efficient Large-scale LLM Training](https://arxiv.org/abs/2505.24034)
- [THUDM/AgentRL](https://github.com/THUDM/AgentRL)
- [Hello from ROLL - ROLL](https://alibaba.github.io/ROLL/)
- [Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning](https://arxiv.org/abs/2504.13914)
- [Building Efficient RL Training for the Agentic Era](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)
- [The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143)
- [NVIDIA/NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner)
- [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [huggingface/trl](https://github.com/huggingface/trl)
- [Accelerate](https://huggingface.co/docs/accelerate)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- [Nsight Systems - NVIDIA Developer](https://developer.nvidia.com/nsight-systems)
- [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [verl-project/verl](https://github.com/verl-project/verl)
- [areal-project/AReaL](https://github.com/areal-project/AReaL)
- [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)
- [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)
- [rllm-org/rllm](https://github.com/rllm-org/rllm)
- [hiyouga/EasyR1](https://github.com/hiyouga/EasyR1)

## 4. Agents and applications

### Agent learning, tools, and trajectory data

- [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789)
- [Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning](https://arxiv.org/abs/2511.14460)
- [Process Reward Models for LLM Agents: Practical Framework and Directions](https://arxiv.org/abs/2502.10325) - AgentPRM: intermediate feedback for agent trajectories.
- [PAIR: Prefix-Aware Internal Reward Model for Multi-Turn Agent Optimization](https://arxiv.org/abs/2605.17877)
- [Agentic Reinforced Policy Optimization](https://arxiv.org/abs/2507.19849) - ARPO: agentic reinforced policy optimization.
- [AEM: Adaptive Entropy Modulation for Multi-Turn Agentic Reinforcement Learning](https://arxiv.org/abs/2605.00425)
- [Closing the Reflection Gap: A Free Calibration Bonus for Agentic RL](https://arxiv.org/abs/2606.14211) - Ref-GRPO: reflection-based calibration.
- [CAPO: Critic-Guided Action-Aligned Policy Optimization for Advancing LLM Agent Capabilities](https://arxiv.org/abs/2604.18401)
- [Turn-PPO: Turn-Level Advantage Estimation with PPO for Improved Multi-Turn RL in Agentic LLMs](https://arxiv.org/abs/2512.17008)
- [AT^2PO: Agentic Turn-based Policy Optimization via Tree Search](https://arxiv.org/abs/2601.04767)
- [AgentR1/Agent-R1](https://github.com/AgentR1/Agent-R1)
- [Unlocking Agentic RL Training for GPT-OSS: A Practical Retrospective](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)
- [Improving Multi-Turn Tool Use with Reinforcement Learning](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)
- [Customizing multiturn AI agents with reinforcement learning](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)
- [IBSEN: Director-Actor Agent Collaboration for Controllable and Interactive Drama Script Generation](https://arxiv.org/abs/2407.01093)
- [Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation](https://arxiv.org/abs/2503.07826)
- [LoopTool: Closing the Data-Training Loop for Robust LLM Tool Calls](https://arxiv.org/abs/2511.09148)
- [Rednote-DeepExperience/LoopTool](https://github.com/Rednote-DeepExperience/LoopTool)
- [From Failure to Mastery: Generating Hard Samples for Tool-use Agents](https://arxiv.org/abs/2601.01498)
- [Bingguang/HardGen · Datasets at Hugging Face](https://huggingface.co/datasets/Bingguang/HardGen)
- [Sample-Efficient Online Learning in LM Agents via Hindsight Trajectory Rewriting](https://arxiv.org/abs/2510.10304)
- [ASTRA: Automated Synthesis of agentic Trajectories and Reinforcement Arenas](https://arxiv.org/abs/2601.21558)
- [LianjiaTech/astra](https://github.com/LianjiaTech/astra)
- [STeCa: Step-level Trajectory Calibration for LLM Agent Learning](https://arxiv.org/abs/2502.14276)
- [WangHanLinHenry/STeCa](https://github.com/WangHanLinHenry/STeCa)
- [TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents](https://arxiv.org/abs/2602.11767)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)
- [VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use](https://arxiv.org/abs/2509.01055)
- [What is the Model Context Protocol (MCP)? - Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro)
- [Tools - Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
- [MCP•RL: Training Agents to Use MCP Servers - ART](https://art.openpipe.ai/features/mcp-rl)
- [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958) - ToolRL: reward-driven tool learning.
- [Agent Workflow Memory](https://arxiv.org/abs/2409.07429)
- [zorazrw/agent-workflow-memory](https://github.com/zorazrw/agent-workflow-memory)
- [rStar2-Agent: Agentic Reasoning Technical Report](https://arxiv.org/abs/2508.20722)
- [Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving](https://arxiv.org/abs/2505.07773)
- [ToRL: Scaling Tool-Integrated RL](https://arxiv.org/abs/2503.23383)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738)
- [rLLM FinQA training account](https://rllm-project.com/post.html?post=finqa.md)
- [rLLM/rLLM-FinQA-4B](https://huggingface.co/rLLM/rLLM-FinQA-4B)
- [FinQA - rLLM](https://docs.rllm-project.com/cookbooks/finqa)

### Search, research, and writing

- [WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents](https://arxiv.org/abs/2509.06501)
- [Tongyi DeepResearch Technical Report](https://arxiv.org/abs/2510.24701)
- [Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities](https://moonshotai.github.io/Kimi-Researcher/)
- [Step-DeepResearch Technical Report](https://arxiv.org/abs/2512.20491)
- [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- [Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](https://arxiv.org/abs/2505.15277) - Web-Shepherd: process rewards for web agents.
- [DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data](https://arxiv.org/abs/2604.19859)
- [Tongyi DeepResearch: A New Era of Open-Source AI Researchers - Tongyi DeepResearch](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)
- [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) - Search-R1: train interleaved reasoning and search.
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470)
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- [DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research](https://arxiv.org/abs/2511.19399)
- [Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards](https://arxiv.org/abs/2601.06021)
- [Agent Reinforcement Fine-Tuning: A Practical Guide (2026)](https://tensorops.ai/blog/practical-guide-to-agent-reinforcement-fine-tuning)
- [Open Deep Research Tutorial - ART](https://art.openpipe.ai/tutorials/open-deep-research)
- [Unlocking the next era of therapeutic discovery: Training an agentic AI researcher with Reinforcement Learning](https://www.owkin.com/blogs-case-studies/unlocking-the-next-era-of-therapeutic-discovery-training-an-agentic-ai-researcher-with-reinforcement-learning)
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160)
- [OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis](https://arxiv.org/abs/2603.20278)
- [PokeeAI/pokee_research_7b](https://huggingface.co/PokeeAI/pokee_research_7b)
- [SFR-DeepResearch: Towards Effective Reinforcement Learning for Autonomously Reasoning Single Agents](https://arxiv.org/abs/2509.06283)
- [Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward](https://arxiv.org/abs/2508.12800)
- [Memento: Fine-tuning LLM Agents without Fine-tuning LLMs](https://arxiv.org/abs/2508.16153)
- [O-Researcher: An Open Ended Deep Research Model via Multi-Agent Distillation and Agentic RL](https://arxiv.org/abs/2601.03743)
- [Fathom-DeepResearch: Unlocking Long Horizon Information Retrieval and Synthesis for SLMs](https://arxiv.org/abs/2509.24107)
- [LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](https://arxiv.org/abs/2506.18841)
- [Writer-R1: Enhancing Generative Writing in LLMs via Memory-augmented Replay Policy Optimization](https://arxiv.org/abs/2603.15061)
- [RL-Struct: A Lightweight Reinforcement Learning Framework for Reliable Structured Output in LLMs](https://arxiv.org/abs/2512.00319)
- [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher)
- [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)
- [stanford-oval/storm](https://github.com/stanford-oval/storm)
- [OPPO-PersonalAI/O-Researcher](https://github.com/OPPO-PersonalAI/O-Researcher)
- [GAIR-NLP/OpenResearcher](https://github.com/GAIR-NLP/OpenResearcher)
- [GAIR-NLP/DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)
- [RUCAIBox/R1-Searcher](https://github.com/RUCAIBox/R1-Searcher)

### Code and software engineering

- [Training Software Engineering Agents and Verifiers with SWE-Gym](https://arxiv.org/abs/2412.21139)
- [ReLook: Vision-Grounded RL with a Multimodal LLM Critic for Agentic Web Coding](https://arxiv.org/abs/2510.11498)
- [Agnostics: Learning to Code in Any Programming Language via Reinforcement with a Universal Learning Environment](https://arxiv.org/abs/2508.04865)
- [Agnostics (ICLR 2026) - ABG](https://abgru.me/project/agnostics/)
- [Agentic Code Reasoning](https://arxiv.org/abs/2603.01896)
- [DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51)
- [DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
- [SWE-smith: Scaling Data for Software Engineering Agents](https://arxiv.org/abs/2504.21798)
- [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](https://arxiv.org/abs/2502.18449)
- [CWM: An Open-Weights LLM for Research on Code Generation with World Models](https://arxiv.org/abs/2510.02387)
- [DeepSWE: Training a Fully Open-sourced, State-of-the-Art Coding Agent by Scaling RL](https://www.together.ai/blog/deepswe)
- [Toward Training Superintelligent Software Agents through Self-Play SWE-RL](https://arxiv.org/abs/2512.18552)
- [Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering](https://arxiv.org/abs/2401.08500)

### Computer use

- [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/abs/2501.12326)
- [bytedance/ui-tars](https://github.com/bytedance/ui-tars)
- [UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2509.02544)
- [Computer use tool - Claude Platform Docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/computer-use-tool)
- [AutoGLM: Autonomous Foundation Agents for GUIs](https://arxiv.org/abs/2411.00820)
- [Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V](https://arxiv.org/abs/2310.11441)
- [MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents](https://arxiv.org/abs/2509.18119)
- [ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents](https://arxiv.org/abs/2508.14040)
- [CogAgent: A Visual Language Model for GUI Agents](https://arxiv.org/abs/2312.08914)
- [zai-org/Open-AutoGLM](https://github.com/zai-org/Open-AutoGLM)

### Business workflows

Direct implementation examples and training accounts. Evaluation criteria below are suggestions for adapting these workflows.

- [AReaL search agent](https://github.com/areal-project/AReaL/tree/main/examples/search_agent) - Search and retrieval. Search environment, tool calls, rollout and reward setup. Suggested evaluation: Answer accuracy, evidence support, calls and cost per solved task.
- [rLLM FinQA](https://docs.rllm-project.com/cookbooks/finqa) - Financial analysis. Financial question answering with a trainable agent. Suggested evaluation: Calculation accuracy, correct evidence, held-out task success.
- [AWS GRPO + TRL walkthrough](https://builder.aws.com/content/35x6VR6kZYSn3JgNQmcNmIVK32Y/fine-tune-small-language-models-for-production-grade-tool-calling-with-grpo-using-hugging-face-trl-on-amazon-sagemaker-ai) - Production tool calling. Small-model tool selection and argument generation. Suggested evaluation: Executable calls, correct arguments, latency and serving cost.
- [Bespoke Labs training account](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning) - Multi-turn API workflows. Environment feedback and multi-turn tool behavior. Suggested evaluation: Task completion and recovery after tool failures.
- [ART Open Deep Research tutorial](https://art.openpipe.ai/tutorials/open-deep-research) - Deep research. Train an agent that researches and synthesizes evidence. Suggested evaluation: Citation support, coverage, completion time and cost.
- [Owkin research-agent training account](https://www.owkin.com/blogs-case-studies/unlocking-the-next-era-of-therapeutic-discovery-training-an-agentic-ai-researcher-with-reinforcement-learning) - Therapeutic discovery. Domain research tasks and feedback. Suggested evaluation: Expert-reviewed correctness and evidence quality.
- [DeepSWE training account](https://www.together.ai/blog/deepswe) - Software maintenance. RL for a coding agent working in software environments. Suggested evaluation: Held-out issue resolution and regression-test success.
- [LinkedIn GPT-OSS retrospective](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl) - Enterprise agent training. Practical agentic RL training and iteration. Suggested evaluation: Domain task success, stability and compute cost.
- [ART MCP RL](https://art.openpipe.ai/features/mcp-rl) - Tool-server integration. Train agents to interact with MCP tools. Suggested evaluation: Valid calls, end-to-end success and permissions compliance.

## 5. Multimodal post-training

### Vision-language models

- [Step3](https://stepfun.ai/research/en/step3)
- [STEP3-VL-10B Technical Report](https://arxiv.org/abs/2601.09668)
- [Xiaomi MiMo-VL-Miloco Technical Report](https://arxiv.org/abs/2512.17436)
- [Kwai Keye-VL Technical Report](https://arxiv.org/abs/2507.01949)
- [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785)
- [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1)
- [QwenLM/Qwen3-VL cookbooks](https://github.com/QwenLM/Qwen3-VL/tree/main/cookbooks)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [Stop Thinking, Start Looking: Efficient Post-Training for Multimodal Document Question Answering via Reasoning-Free Alignment](https://arxiv.org/abs/2607.14682)

### Speech and audio

- [Step-Audio-R1 Technical Report](https://arxiv.org/abs/2511.15848)
- [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312)
- [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)
- [SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models](https://arxiv.org/abs/2308.16692)
- [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)
- [Step-Audio 2 Technical Report](https://arxiv.org/abs/2507.16632)
- [Mind-Paced Speaking: A Dual-Brain Approach to Real-Time Reasoning in Spoken Language Models](https://arxiv.org/abs/2510.09592)
- [Step-Audio-R1.5 Technical Report](https://arxiv.org/abs/2604.25719)
- [stepfun-ai/Step-Audio-R1](https://github.com/stepfun-ai/Step-Audio-R1)
- [GPT-4o System Card](https://arxiv.org/abs/2410.21276)

### Image and video generation

- [DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818) - DanceGRPO: group-relative optimization for visual generation.
- [Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation - Stability AI](https://stability.ai/research/pick-a-pic)
- [LAION-Aesthetics - LAION](https://laion.ai/blog/laion-aesthetics/)
- [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301) - DDPO: reinforcement learning for diffusion models.
- [Simple statistical gradient-following algorithms for connectionist reinforcement learning - Machine Learning - Springer Nature Link](https://doi.org/10.1007/BF00992696)
- [DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2305.16381)
- [Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation](https://arxiv.org/abs/2305.01569)
- [Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis](https://arxiv.org/abs/2306.09341)
- [Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400) - AlignProp: optimize differentiable rewards.
- [Video Diffusion Alignment via Reward Gradients](https://arxiv.org/abs/2407.08737) - VADER: video alignment via reward gradients.
- [XueZeyue/DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
- [Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning](https://arxiv.org/abs/2311.10709)
- [NeurIPS Poster Boosting Text-to-Video Generative Model with MLLMs Feedback](https://neurips.cc/virtual/2024/poster/96722)
- [Seedance 1.0: Exploring the Boundaries of Video Generation Models](https://arxiv.org/abs/2506.09113)
- [LongCat-Video Technical Report](https://arxiv.org/abs/2510.22200)
- [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314)
- [Veo 3.1 - Google DeepMind](https://deepmind.google/models/veo/)

## 6. Evaluation and alignment

### Benchmarks and evaluation

- [modelscope/evalscope](https://github.com/modelscope/evalscope)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
- [Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces](https://arxiv.org/abs/2601.11868)
- [LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code](https://arxiv.org/abs/2403.07974)
- [CyberGym: Evaluating AI Agents' Real-World Cybersecurity Capabilities at Scale](https://arxiv.org/abs/2506.02548)
- [The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models](https://proceedings.mlr.press/v267/patil25a.html)
- [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)
- [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854)
- [VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks](https://arxiv.org/abs/2401.13649)
- [Vending-Bench: A Benchmark for Long-Term Coherence of Autonomous Agents](https://arxiv.org/abs/2502.15840)
- [GAIA: a benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983)
- [Mind2Web: Towards a Generalist Agent for the Web](https://arxiv.org/abs/2306.06070)
- [AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents](https://arxiv.org/abs/2407.18901)
- [Berkeley Function Calling Leaderboard (BFCL) V4](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [ACEBench: Who Wins the Match Point in Tool Usage?](https://arxiv.org/abs/2501.12851)
- [API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs](https://arxiv.org/abs/2304.08244)
- [The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](https://arxiv.org/abs/2510.25726)
- [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://arxiv.org/abs/2404.07972)
- [AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents](https://arxiv.org/abs/2405.14573)
- [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355)
- [Object Hallucination in Image Captioning](https://arxiv.org/abs/1809.02156)

### Safety and instruction hierarchy

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Collective Constitutional AI](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input)
- [Claude's constitution](https://www.anthropic.com/research/claudes-constitution)
- [Claude's new constitution](https://www.anthropic.com/news/claude-new-constitution)
- [The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions](https://openai.com/index/the-instruction-hierarchy/)
- [Natural Emergent Misalignment from Reward Hacking in Production RL](https://arxiv.org/abs/2511.18397)
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://www.anthropic.com/news/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training)
- [Constitutional AI: Harmlessness from AI feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
- [InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents](https://arxiv.org/abs/2403.02691)
- [AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents](https://arxiv.org/abs/2406.13352)
- [Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents](https://arxiv.org/abs/2410.02644)
- [EVA: Evolving Semantic Adversaries for Red-Teaming GUI Agents Against Environmental Injection Attacks](https://arxiv.org/abs/2505.14289)
- [The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions](https://arxiv.org/abs/2404.13208)
- [Improving instruction hierarchy in frontier LLMs](https://openai.com/index/instruction-hierarchy-challenge/)
- [Constitutional AI: Harmlessness from AI feedback](https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback)
- [Anthropic’s Responsible Scaling Policy](https://www.anthropic.com/responsible-scaling-policy)

## Frontier

Original model reports and development accounts. Disclosure varies across sources; use the core sections for algorithm and implementation references.

### Model reports and training accounts

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [MiniMax M2.1: Post-Training Experience and Insights for Agent Models](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement](https://arxiv.org/abs/2409.12122)
- [QwQ-32B: Embracing the Power of Reinforcement Learning](https://qwenlm.github.io/blog/qwq-32b/)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3-Coder: Agentic Coding in the World](https://qwenlm.github.io/blog/qwen3-coder/)
- [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729)
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
- [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)
- [Seed-Prover 1.5](https://seed.bytedance.com/en/blog/seed-prover-1-5-advanced-mathematical-reasoning-through-a-novel-agentic-architecture)
- [Seed1.8](https://seed.bytedance.com/en/blog/official-release-of-seed1-8-a-generalized-agentic-model)
- [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471)
- [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)
- [llm.hunyuan.T1](https://tencent.github.io/llm.hunyuan.T1/README_EN.html)
- [Tencent-Hunyuan/Hunyuan-A13B](https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)
- [tencent/Hunyuan-A13B-Instruct](https://huggingface.co/tencent/Hunyuan-A13B-Instruct)
- [ERNIE 5.0 Technical Report](https://arxiv.org/abs/2602.04705)
- [LongCat-Flash-Thinking-2601 training report](https://tech.meituan.com/2026/02/02/LongCat-Flash-Thinking-2601-techreport.html)
- [Ling-1T development account](https://ant-ling.medium.com/deep-insight-efficient-inference-introducing-the-trillion-parameter-ling-1t-model-77d6170e5e8e)
- [Ring-1T development account](https://ant-ling.medium.com/ring-1t-release-the-flow-state-of-insight-born-of-epiphany-c20e8e32817c)
- [pangu-tech/pangu-ultra](https://github.com/pangu-tech/pangu-ultra)
- [Pangu Pro MoE: Mixture of Grouped Experts for Efficient Sparsity](https://arxiv.org/abs/2505.21411)
- [Yi-Lightning Technical Report](https://arxiv.org/abs/2412.01253)
- [InternLM2 Technical Report](https://arxiv.org/abs/2403.17297)
- [Baichuan 2: Open Large-scale Language Models](https://arxiv.org/abs/2309.10305)
- [360Zhinao Technical Report](https://arxiv.org/abs/2405.13386)
- [Skywork Open Reasoner 1 Technical Report](https://arxiv.org/abs/2505.22312)
- [MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining](https://arxiv.org/abs/2505.07608)
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
- [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/)
- [OpenAI o3 and o4-mini System Card](https://openai.com/index/o3-o4-mini-system-card/)
- [Addendum to OpenAI o3 and o4-mini system card: OpenAI o3 Operator](https://openai.com/index/o3-o4-mini-system-card-addendum-operator-o3/)
- [GPT-4.5 system card](https://openai.com/index/gpt-4-5-system-card/)
- [GPT-5 System Card](https://openai.com/index/gpt-5-system-card/)
- [GPT-5.1 Instant and GPT-5.1 Thinking System Card Addendum](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-1/)
- [GPT-5.4 Thinking System Card](https://openai.com/index/gpt-5-4-thinking-system-card/)
- [GPT-5.5 System Card](https://openai.com/index/gpt-5-5-system-card/)
- [GPT-5.5 Instant System Card](https://openai.com/index/gpt-5-5-instant-system-card/)
- [Addendum to GPT-5 system card: GPT-5-Codex](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-codex/)
- [GPT-5.1-Codex-Max System Card](https://openai.com/index/gpt-5-1-codex-max-system-card/)
- [Introducing GPT-5.2-Codex](https://openai.com/index/introducing-gpt-5-2-codex/)
- [Claude 4 System Card](https://www.anthropic.com/claude-4-system-card)
- [Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context](https://arxiv.org/abs/2403.05530)
- [Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities](https://arxiv.org/abs/2507.06261)
- [Gemini 2.5: Deep Think is now rolling out](https://blog.google/products/gemini/gemini-2-5-deep-think)
- [Introducing the Gemini 2.5 Computer Use model](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-computer-use-model/)
- [Gemini 3.1 Pro - Model Card - Google DeepMind](https://deepmind.google/models/model-cards/gemini-3-1-pro/)
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905)
- [Phi-4-reasoning Technical Report](https://arxiv.org/abs/2504.21318)
- [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704)
- [Llama-Nemotron: Efficient Reasoning Models](https://arxiv.org/abs/2505.00949)
- [NVIDIA Llama Nemotron Ultra Open Model Delivers Groundbreaking Reasoning Accuracy](https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra-open-model-delivers-groundbreaking-reasoning-accuracy/)
- [Build Enterprise AI Agents with Advanced Open NVIDIA Llama Nemotron Reasoning Models](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/)
- [Introducing the Nemotron-H Reasoning Model Family: Throughput Gains Without Compromise](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/)
- [Inside NVIDIA Nemotron 3: Techniques, Tools, and Data That Make It Efficient and Accurate](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/)
- [Magistral](https://arxiv.org/abs/2506.10910)
- [Apple Intelligence Foundation Language Models - Apple Machine Learning Research](https://machinelearning.apple.com/research/apple-intelligence-foundation-language-models)
- [Apple Intelligence Foundation Language Models Tech Report 2025 - Apple Machine Learning Research](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)
- [Grok-1 Model Card - SpaceXAI](https://x.ai/news/grok/model-card)
- [IBM Granite 3.3: Speech recognition, refined reasoning, and RAG LoRAs](https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras)
- [IBM Granite 4.0: Hyper-efficient, High Performance Hybrid Models for Enterprise](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)
- [Granite 4.1 LLMs: How They’re Built](https://huggingface.co/blog/ibm-granite/granite-4-1)
- [xLAM: A Family of Large Action Models for AI Agents](https://www.salesforce.com/blog/large-action-model-ai-agent/)
- [The amazon nova family of models: Technical report and model card](https://www.isi.edu/results/publications/31887/the-amazon-nova-family-of-models-technical-report-and-model-card/)
- [Amazon Nova Premier: Technical report and model card](https://www.amazon.science/publications/amazon-nova-premier-technical-report-and-model-card)
- [Jamba 1.5a: Enhancing AI Safety Through Post-Post-Training Alignment - AI21](https://www.ai21.com/research/jamba-1-5a/)
- [Jamba 1.5a Whitepaper_April 2025 - Google Docs](https://lp.ai21.com/hubfs/resources/Jamba-1-5a-Whitepaper.pdf)
- [A technical report on Composer 2 · Cursor](https://cursor.com/blog/composer-2-technical-report)
- [HyperCLOVA X Technical Report](https://arxiv.org/abs/2404.01954)
- [HyperCLOVA X THINK Technical Report](https://arxiv.org/abs/2506.22403)
- [Tülu 3: The next era in open post-training - Ai2](https://allenai.org/blog/tulu-3-technical)

## Source policy

A core source must be one of the following:

- the paper that introduced the mechanism;
- the official documentation that defines it;
- the creator repository that implements it;
- a direct implementer report with code, measurements, or enough detail to understand the training setup.

Reported improvements need the model, data, reward, baseline, and evaluation method. Business use cases should distinguish measured results from suggested applications.

See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a resource.
