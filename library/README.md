# Harness Library

This directory contains Turin's serious, ready-to-use harness library.

It is organized into:

- `library/blocks/` — reusable harness building blocks
- `library/workflows/` — full end-to-end harness workflows

These are not tutorial snippets. They are intended to be:

- practical
- adoptable
- validated by tests
- strong starting points for real usage

Validation:

```bash
cargo test --test example_harness_examples
```

Current highlighted entries:

- `library/workflows/openclaw_style_personal_assistant/` — workspace-contract personal assistant with planner/reviewer routing and durable artifacts
- `library/workflows/full_coding_harness/` — spec/task-driven coding workflow with planner + reviewer specialists
- `library/workflows/bug_triage_desk/` — issue-intake workflow with triager + responder specialists
- `library/workflows/release_manager/` — release-readiness workflow with review + changelog drafting
- `library/blocks/code_reviewer/` — focused review contract for correctness/regression review
- `library/blocks/task_planner/` — focused planning contract for sequenced task breakdowns
- `library/blocks/spec_writer/` — focused contract for turning rough ideas into executable specs
- `library/blocks/test_gap_finder/` — focused contract for identifying likely missing tests
- `library/blocks/repo_librarian/` — focused contract for repository-aware routing and guidance
- `library/blocks/governed_peer_review/` — temporary-grant peer review
- `library/blocks/delegated_peer_capabilities/` — delegated-capability peer completion
- `library/blocks/durable_journal/` — durable runtime DB journaling
