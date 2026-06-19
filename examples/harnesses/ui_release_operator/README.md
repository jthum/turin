# UI Release Operator Harness

This harness is a UI contract fixture for Turin clients. It declares a small
release-operations app with menus, multiple screens, worklist-backed lists,
worklist-backed activity/detail surfaces, normal actions, confirmed actions, a
form node with text/select/number/decimal fields, a diagnostic failure drill,
worklist-backed report/chart surfaces, pane overlays, dynamic badges, and
open/show/focus/refresh hints.

The harness does not mutate durable state during load. Use the rendered app's
`Seed Demo Work` action or the Intake form to create release work items, then use
the confirmed actions to approve or reject the next pending approval. Clients
should refresh the visible `worklists.release` lists after those actions
complete.

`Run Failure Drill` intentionally raises a Lua error. It exists so UI clients can
exercise failed action envelopes, notices, and action-result panels against a
declared action rather than an undeclared typo.
