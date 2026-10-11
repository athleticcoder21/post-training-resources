# Contributing

This repository indexes the core technology and practical applications of post-training: algorithms, papers, systems, data, benchmarks, model reports, and applied recipes.

## Adding a resource

The main sections follow this order: training infrastructure, continual pretraining and domain knowledge, imitation learning, preference optimization, on-policy distillation, RL (primarily RLVR), agentic RL, and multimodal RL. Infrastructure should lead with educational explanations and worked examples, with concise framework descriptions as implementation companions. Put environment design, rewards, harnesses, and sandbox execution in the RL environments and harnesses subsection. Put RL rollout architecture, async training, and weight synchronization inside the RL section. Place a framework link beside the training method it implements. Distinguish corpus-based CPT, teacher-solution SFT before RL, and on-policy distillation. Keep evaluation and model reports in their reference sections.

1. Choose the closest topic in the README and add a bullet at the right point in its reading order. Put introductions and guided examples before specialized objectives, theoretical analyses, or large-scale systems; use prerequisites rather than publication date.
2. Link directly to the original paper, author repository, dataset, model card, documentation, or practitioner account.
3. Use a concise linked title followed by a short description, matching the README: `- [Resource](https://example.com) - What it contributes.`
4. Add a short explanation when the title is not enough. For algorithms, name the objective or method; for systems, explain their training role.
5. Check existing entries to avoid duplicate papers, alternate arXiv formats, and repeated repository links.
6. Open the cited paper, documentation, or creator repository. Verify the exact title, method, training objective, and whether it updates weights or only changes inference. Check that the linked repository belongs to the cited work. Verify publication status before calling a paper peer-reviewed; label unverified manuscripts as preprints. Do not infer relevance from a title or copy an unverified citation from another index.
7. Prefer one authoritative link per idea. Add a paper and code companion only when the code is the implementation and adds practical detail. Exclude product announcements, inference-only prompting, and loosely related papers from training sections.
8. Run `python3 scripts/check_guide.py`.

## Adding an applied recipe

Include the task or business workflow, an original implementation or training account, the feedback or reward used, and the evaluation setup. Label suggested business uses as suggestions. Reported results should identify their model, data, baseline, and evaluation conditions.

Distinguish executable training code, inference examples, simulations, and production case studies. A paper does not need a business use case to belong in this index.

## Assets and pull requests

Keep images and visual assets in `assets/`. Use concise descriptions, preserve credit to the original authors, and disclose authorship or financial interests.

Use the pull request template. Suggestions, broken links, and inaccurate descriptions can be reported using the issue templates.
