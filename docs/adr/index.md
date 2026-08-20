# Architecture Decision Records

This directory captures the runtime and daemon decisions that are easy to miss if you only read the code.

Current ADRs:

- `001-execution-host-split.md` — why execution moved out of `Kernel`
- `002-filesystem-authoritative-daemon-state.md` — why daemon state lives in the filesystem, not a hidden registry
- `003-typed-ndjson-daemon-protocol.md` — why the daemon protocol is typed NDJSON over a local socket
- `004-harness-engine-sync-mutex.md` — why harness engine access remains sync and mutex-guarded
- `005-cooperative-and-forceful-cancellation.md` — why Turin exposes both cancel and kill semantics
- `006-trace-id-task-lineage.md` — why task lineage is explicit and survives peer-agent hops
- `007-rust-harness-boundary.md` — why Rust harness policy does not receive a bag of runtime managers

Status values used here:

- `Accepted` — current implemented decision
- `Superseded` — historical decision replaced by a later ADR
