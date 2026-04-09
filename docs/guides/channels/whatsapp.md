# WhatsApp Channel Setup

This guide is the practical path from "I want Turin on WhatsApp" to "my WhatsApp channel runtime is linked and responding."

It complements:

- `docs/operations/daemon.md` for the daemon and channel reference
- `docs/reference/channel-sidecars.md` for the shared sidecar/auth-flow protocol
- `examples/config/channels/whatsapp-channel.toml.example` for a ready-to-adapt channel config

## What The Current WhatsApp Adapter Supports

Current scope:

- WhatsApp linked-device login through QR pairing
- headless pairing-code login when a phone number is provided
- inbound text messages
- inbound media downloads into Turin-managed local storage
- outbound text replies and local file uploads
- direct messages and group chats
- `user` and `room` session scopes
- personal-account safety defaults

Current non-goals:

- WhatsApp Cloud API
- threads
- streaming previews

This adapter is for the linked-device model. If Turin later adds an official Cloud API adapter, it should be a separate channel kind such as `whatsapp-api`.

## 1. Choose Account Mode

WhatsApp now supports two operating modes:

- `personal`
- `dedicated`

### Dedicated account

Use this when the WhatsApp number belongs only to the agent.

This is the cleanest operational model:

- the account does not mix personal traffic with agent traffic
- you usually do not need a trigger prefix
- a full chat or group can be treated as agent space

Recommended defaults:

- `account_mode = "dedicated"`
- `pairing_mode = "auto"` or `pending`
- `session_scope = "room"` for a shared group, or `user` for separate per-sender sessions

### Personal account

Use this when Turin is linked to your own WhatsApp account.

This is supported, but it needs stricter defaults so Turin does not intrude on normal conversations.

Current personal-account behavior:

- self-originated messages are ignored
- personal mode defaults `trigger_prefix` to `/turin`
- only messages with the prefix are treated as agent input unless you change the setting

Recommended operating model:

- keep self-note chat for normal personal use
- create a private group for Turin interaction
- optionally restrict Turin further with `allowed_chats`

If you link Turin to your personal account, a private approved group is the intended interaction surface. Self-chat is not the recommended control surface.

## 2. Pick A Pairing Method

The adapter supports two setup flows:

- QR pairing
- pairing code for headless servers

### QR pairing

This is the default path.

Turin starts a temporary auth flow, renders a QR code, and waits for you to link the account from the WhatsApp mobile app.

### Pairing code

This is for headless servers where rendering a QR code is inconvenient.

Set:

- `pair_code_phone_number`

Optional:

- `pair_code_custom_code`

The pairing phone number should be an international number string. The custom code must be 8 Crockford Base32 characters. Both fields are cleared after pairing completes.

## 3. Session Scope

WhatsApp supports:

- `user`
- `room`

Meaning:

- `user`: each sender gets an independent Turin session inside a chat
- `room`: everyone in the same chat shares one Turin session

Practical recommendations:

- personal account in a private group: `user`
- dedicated team bot in one group: `room`
- direct one-on-one assistant: `user`

## 4. Guardrails For Personal Accounts

The main guardrails are:

- `trigger_prefix`
- `allowed_chats`
- `banned_chats`
- shared runner access control:
  - `pairing_mode`
  - `pairing_users`
  - `allowed_users`
  - `banned_users`

Recommended personal-account baseline:

- `account_mode = "personal"`
- `trigger_prefix = "/turin"`
- `pairing_mode = "pending"` if you want explicit approval
- `allowed_chats = [...]` if only a small set of chats should ever trigger Turin

Recommended dedicated-account baseline:

- `account_mode = "dedicated"`
- empty `trigger_prefix`
- `pairing_mode = "auto"` or explicit approved chat policy

`banned_chats` always wins over `allowed_chats`.

## 5. Example Config

Start from:

- `examples/config/channels/whatsapp-channel.toml.example`

Typical personal-account config:

```toml
enabled = true
kind = "whatsapp"
agent_id = "default"

account_mode = "personal"
workspace_id = "whatsapp"
pairing_mode = "pending"
session_scope = "user"
trigger_prefix = "/turin"

# allowed_chats = ["120363400000000000@g.us"]
# banned_chats = ["status@broadcast"]
```

Typical dedicated-account config:

```toml
enabled = true
kind = "whatsapp"
agent_id = "default"

account_mode = "dedicated"
workspace_id = "whatsapp"
pairing_mode = "auto"
session_scope = "room"
```

## 6. Run The Setup Flow

If you use `turin-manager`, the WhatsApp sidecar exposes the auth flow through the shared sidecar manifest. That means the manager can drive QR pairing or pairing-code setup without a WhatsApp-specific manager implementation.

If you are using a headless machine:

- set `pair_code_phone_number`
- run the setup flow
- enter the generated code in WhatsApp on the phone

On successful pairing, Turin writes the linked-device session path back into:

- `session_store_path`

You normally do not need to set that path yourself unless you want to pin it somewhere specific.

## 7. Operational Notes

- Inbound media is downloaded into Turin-managed local storage and forwarded as attachment refs.
- Outbound media currently requires `local_path`; plain remote URLs are not uploaded directly.
- Self-originated messages are ignored to avoid obvious self-reply loops.
- Personal mode is intentionally conservative; if you remove the trigger prefix, Turin becomes much easier to trigger accidentally.
- If you want the cleanest experience, use a dedicated number. If you want the most convenient personal setup, use a private group with the default `/turin` prefix.
