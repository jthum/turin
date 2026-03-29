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
- the target room ID
- the auth token

The token is typically written to a `.env` file next to `turin.toml`, and Turin will load that adjacent `.env` automatically on startup.

## Settings

Common settings:

- `base_url`
- `user_id`
- `room_id`
- `token_env`
- `respond_mode = "all" | "mentions"`
- `session_scope = "user" | "thread" | "room"`

Advanced settings:

- `allowed_users`
- `banned_users`
- `poll_interval_ms`
- `start_from_latest`
- `ignore_bot_messages`

## Runtime Notes

- direct messages are always accepted
- shared rooms can be limited to mentions-only mode
- `session_scope = "thread"` replies in Rocket.Chat threads by setting `tmid`
- the first implementation uses REST polling rather than the Rocket.Chat realtime websocket stack
