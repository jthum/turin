# UI Release Operator Harness

This harness demonstrates a focused release-operations app for Turin clients. It
declares menus, multiple screens, worklist-backed lists, activity/detail
surfaces, normal actions, confirmed actions, a form with text/select/number/
decimal fields, report/chart surfaces, pane overlays, dynamic badges, and
open/show/focus/refresh hints.

The harness does not mutate durable state during load. Use the rendered app's
`Seed Approval Work` action or the Intake form to create release work items, then
use the confirmed actions to approve or reject the next pending approval. Clients
should refresh the visible `worklists.release` lists after those actions
complete.

The harness also declares `release.fail_diagnostic` as a hidden test hook so
client and web tests can exercise failed action envelopes against a declared
action. It is not shown in the default app surface.
