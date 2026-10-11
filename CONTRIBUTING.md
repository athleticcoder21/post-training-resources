# Contributing

This repository indexes the core technology and practical applications of post-training: algorithms, papers, systems, data, benchmarks, model reports, and applied recipes.

## Adding a resource

1. Choose the closest topic in the README and add a bullet to its resource list.
2. Link directly to the original paper, author repository, dataset, model card, documentation, or practitioner account.
3. Use a concise linked title followed by a short description, matching the README: `- [Resource](https://example.com) - What it contributes.`
4. Add a short explanation when the title is not enough. For algorithms, name the objective or method; for systems, explain their training role.
5. Check existing entries to avoid duplicate papers, alternate arXiv formats, and repeated repository links.
6. Open the original source and verify its title and relevance. Do not copy an unverified citation from another index.
7. Run `python3 scripts/check_guide.py`.

## Adding an applied recipe

Include the task or business workflow, an original implementation or training account, the feedback or reward used, and the evaluation setup. Label suggested business uses as suggestions. Reported results should identify their model, data, baseline, and evaluation conditions.

Distinguish executable training code, inference examples, simulations, and production case studies. A paper does not need a business use case to belong in this index.

## Assets and pull requests

Keep images and visual assets in `assets/`. Use concise descriptions, preserve credit to the original authors, and disclose authorship or financial interests.

Use the pull request template. Suggestions, broken links, and inaccurate descriptions can be reported using the issue templates.
