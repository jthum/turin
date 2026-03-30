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
- `dm_session_scope = "user" | "thread" | "room"` (optional override)
- `reply_mode = "thread" | "channel" | "thread_and_channel"`
- `stream_mode = "off" | "typing"`

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
- `persist_thinking`

## Runtime Notes

- direct messages bypass mention checks, but still respect room approval and user allow/ban rules
- shared rooms can be limited to mentions-only mode
- `reply_mode = "thread"` replies in Rocket.Chat threads by setting `tmid`
- `reply_mode = "thread_and_channel"` posts into the thread and also shows the reply in the room with Rocket.Chat's `tshow`
- `reply_mode = "channel"` posts directly in the room and includes the triggering message as a Rocket.Chat attachment-style quote instead of starting a thread
- `dm_session_scope = "room"` is the practical choice if you want direct messages to continue in one session while shared rooms stay per thread
- once Turin has replied in a thread, subsequent messages in that same thread are accepted without mentioning the bot again
- quoting a recent Turin message in the room is also accepted as a follow-up trigger, even without another mention
- `stream_mode = "typing"` sends Rocket.Chat room activity notifications while a turn is running
- `persist_thinking = true` prepends final model thinking to the posted reply
- markdown pipe tables are wrapped in fenced code blocks automatically so they stay readable in Rocket.Chat
- pairing and room approval use the same generic Turin access-state model as Telegram
- realtime websocket/DDP inbound delivery is the default
- `transport_mode = "polling"` remains available as a fallback if the server's realtime path is unavailable
