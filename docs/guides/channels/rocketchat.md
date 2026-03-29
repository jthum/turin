# Rocket.Chat

Rocket.Chat support is provided by the external `turin-channel-rocketchat` sidecar.

## Setup

Use `turin-manager` to configure the channel:

```bash
turin-manager init
turin-manager channels list
turin-manager channels configure rocketchat
turin-manager channels status
turin-manager doctor
```

The manager asks for:

- the Rocket.Chat server URL
- the API user ID
- the auth token
- the pairing mode

By default, Rocket.Chat now behaves like Telegram:

- new rooms and DMs can be discovered dynamically
- `pairing_mode` controls whether newly seen rooms are auto-approved, held for approval, or rejected
- `pairing_users` can restrict who is allowed to introduce a new room
- once a room is approved, `allowed_users` and `banned_users` still apply inside that room

If you want a static setup instead, set an optional `room_id` or `room_name` filter to pin the channel to one specific room or DM.

The token is typically written to a `.env` file next to `turin.toml`, and Turin will load that adjacent `.env` automatically on startup.

## Settings

Common settings:

- `base_url`
- `user_id`
- `token_env`
- `pairing_mode = "auto" | "pending" | "off"`
- `transport_mode = "realtime" | "polling"`
- `respond_mode = "all" | "mentions"`
- `session_scope = "user" | "thread" | "room"`

Advanced settings:

- `workspace_id`
- `pairing_users`
- `websocket_url`
- `room_id`
- `room_name`
- `allowed_users`
- `banned_users`
- `poll_interval_ms`
- `start_from_latest`
- `ignore_bot_messages`

## Runtime Notes

- direct messages bypass mention checks, but still respect room approval and user allow/ban rules
- shared rooms can be limited to mentions-only mode
- `session_scope = "thread"` replies in Rocket.Chat threads by setting `tmid`
- pairing and room approval use the same generic Turin access-state model as Telegram
- realtime websocket/DDP inbound delivery is the default
- `transport_mode = "polling"` remains available as a fallback if the server's realtime path is unavailable
