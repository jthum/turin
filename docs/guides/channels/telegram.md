# Telegram Channel Setup

This guide is the practical path from "I want Turin in Telegram" to "my Telegram channel runtime is online and responding."

It complements:

- `docs/operations/daemon.md` for the full daemon/channel reference
- `docs/operations/live-provider-testing.md` for the live smoke path
- `scripts/live_telegram_channel_smoke.sh` for a runnable validation script

## What The First Telegram Adapter Supports

Current Phase 8 scope:

- Telegram Bot API via long polling
- inbound text messages
- outbound replies with automatic reply threading to the source message
- Telegram HTML rendering for code blocks
- deterministic routing by Telegram chat and forum topic
- one Turin Telegram channel can watch one or many Telegram chats
- generic pairing modes for chat discovery (`off`, `pending`, `auto`)
- generic sender access policies (`pairing_users`, `allowed_users`, `banned_users`)
- configurable group trigger policy (`all`, `mentions`, `replies`, `mentions_or_replies`)
- daemon-managed lifecycle (`create`, `enable`, `disable`, `update`, `status`)

Current non-goals:

- webhooks
- rich media
- inline keyboards and advanced Telegram UI

## 1. Create a Bot

Create a bot with BotFather:

1. Open BotFather in Telegram.
2. Run `/newbot`.
3. Pick a display name and username.
4. Copy the bot token.

Store the token in an environment variable:

```bash
export TELEGRAM_BOT_TOKEN="123456789:replace-me"
```

Turin reads the token from `token_env`, so you can use a different env var name if you prefer.

If you want the setup wizard path instead of hand-editing files:

```bash
turin-manager init
turin-manager channels list
turin-manager channels configure telegram
turin-manager channels status
turin-manager doctor
```

`turin-manager channels configure telegram` validates the token, stages the resulting `channel.toml` diff, and can write the token into a `.env` file next to `turin.toml`. Turin loads that adjacent `.env` automatically on startup.

## 2. Decide Which Telegram Surfaces You Want

Turin’s `chat_id` is a numeric Telegram chat identifier. A Telegram channel can watch:

- one chat via `chat_id`
- many chats via `chat_ids`
- zero preconfigured chats when pairing mode is enabled

That means you do not need one Turin channel per Telegram group unless you want different agent/harness settings per group.

The setup differs slightly by chat type.

### Direct bot chat

Send a message to the bot directly so Telegram creates the conversation.

### Group or supergroup

Add the bot to the group and send a test message.

If you want the bot to receive ordinary group messages, disable privacy mode in BotFather:

1. Run `/setprivacy`.
2. Select the bot.
3. Choose `Disable`.

If privacy mode stays enabled, Telegram may only deliver commands, replies, and mentions instead of normal text traffic.

Turin can also enforce its own group trigger policy with `respond_mode`:

- `all`: respond to any non-empty message in configured non-private chats
- `mentions`: respond only when the bot is explicitly mentioned
- `replies`: respond only when a message replies to the bot
- `mentions_or_replies`: respond when either of the above is true

For shared groups, `mentions_or_replies` is usually the right setting.

### Channel

Add the bot as an admin in the channel and send a test post.

Turin can receive `channel_post` updates, but the bot must actually be allowed to observe that channel.

### Forum topics in supergroups

Turin uses the same `chat_id` for the whole supergroup and automatically routes by Telegram `message_thread_id` when messages arrive inside a forum topic.

That means:

- one supergroup can host multiple stable Turin conversations
- each topic gets its own deterministic Turin slot

## 3. Remove Any Active Webhook

The current Turin Telegram adapter is long-polling only.

If the bot already has a webhook configured, `getUpdates` will fail until you remove it:

```bash
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/deleteWebhook?drop_pending_updates=true"
```

You can also inspect current webhook state:

```bash
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo" | jq
```

If `url` is non-empty, the bot is still configured for webhook delivery.

## 4. Discover The Numeric `chat_id`

If you are using explicit allowlisting with `chat_id` or `chat_ids`, you need the numeric ids up front.

If you are using pairing mode instead:

- `pairing_mode = "auto"` lets Turin discover new chats automatically
- `pairing_mode = "pending"` records unknown chats and holds them for operator approval
- with either pairing mode, you can omit `chat_id` and `chat_ids`

After sending a test message or post in the target chat, fetch recent updates:

```bash
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getUpdates" | jq
```

Look for one of these payload shapes:

- direct/group messages: `.result[].message.chat.id`
- channel posts: `.result[].channel_post.chat.id`

For forum topics, also note:

- `.result[].message.message_thread_id`

Typical values:

- direct chats: positive integer
- groups/channels/supergroups: negative integer, often starting with `-100`

If you want one Turin Telegram channel to handle several groups plus your DM, collect all of those ids and use `chat_ids`.

If you do not see the chat you expect:

- make sure you sent a fresh test message after adding the bot
- make sure privacy mode is disabled for normal group traffic
- make sure the bot is an admin if you are targeting a channel
- make sure no webhook is still active

## 5. Pairing And Sender Access Control

Telegram chat ids solve explicit room allowlisting, but they are not a good general-user onboarding path on their own. Turin now supports two generic alternatives through the shared channel runner:

- `pairing_mode = "auto"`: unknown rooms are approved automatically the first time an eligible sender reaches them
- `pairing_mode = "pending"`: unknown rooms are recorded as pending and Turin replies with a pending-approval notice until the operator approves them

Sender access is split into three separate concerns:

- `pairing_users`: who is allowed to auto-pair or register new rooms when `pairing_mode` is `auto` or `pending`
- `allowed_users`: who may interact inside already-approved rooms; when omitted or empty, any sender in an approved room may interact
- `banned_users`: explicit deny list; this overrides `allowed_users`

For Telegram, selectors are simple strings interpreted by the Telegram adapter:

- numeric string: matched against Telegram user id
- non-numeric string: matched against Telegram username
- `@username` also works as a convenience form

Telegram Bot API does not expose private email addresses or phone numbers for arbitrary users, so those are not realistic selector types here.

If you enable `pairing_mode = "auto"` without `pairing_users`, Turin will approve the first sender who reaches a new room. That is convenient for private testing and risky for public groups.

Practical recommendations:

- personal bot across several groups:
  - `pairing_mode = "auto"`
  - `pairing_users = [498502840]`
  - `respond_mode = "mentions_or_replies"`
- personal bot that only you can pair, but everyone in approved rooms can use:
  - `pairing_mode = "auto"`
  - `pairing_users = [498502840]`
  - omit `allowed_users`
  - `respond_mode = "mentions_or_replies"`
- shared bot in a controlled team group:
  - explicit `chat_ids`
  - `respond_mode = "mentions_or_replies"`
- cautious rollout into unknown groups:
  - `pairing_mode = "pending"`
  - then inspect/approve with `turin daemon channel access ...`

## 6. Session Scope

Telegram now supports configurable session scopes:

- `user`: default; each sender gets an independent session inside the chat or topic
- `thread`: one shared session per Telegram topic/thread; all senders in that thread share context
- `room`: one shared session for the whole chat; all senders and all topics in that chat share context

This setting controls session routing, not room approval:

- pairing and access control still happen at the room level
- session scope only changes which inbound messages reuse the same Turin session

When `session_scope` is `thread` or `room`, Turin adds sender attribution to the prompt before it reaches the model, so the shared session still knows who said what.

Practical recommendations:

- personal bot in a group: `session_scope = "user"`
- shared bot per topic/forum thread: `session_scope = "thread"`
- one shared bot for a small group chat: `session_scope = "room"`

## 7. Create The Turin Channel

Create the channel with the daemon CLI:

```bash
turin daemon channel create telegram-ops \
  --kind telegram \
  --agent default \
  --setting token_env=TELEGRAM_BOT_TOKEN \
  --setting chat_id=-1001234567890
```

Useful optional settings:

```bash
turin daemon channel create telegram-ops \
  --kind telegram \
  --agent default \
  --setting token_env=TELEGRAM_BOT_TOKEN \
  --setting chat_ids=-1001234567890,-100987654321,498502840 \
  --setting poll_timeout_secs=10 \
  --setting poll_interval_ms=250 \
  --setting respond_mode=mentions_or_replies \
  --setting session_scope=user \
  --setting pairing_mode=auto \
  --setting pairing_users=498502840 \
  --setting stream_mode=block \
  --setting stream_thinking=false \
  --setting persist_thinking=false \
  --setting start_from_latest=true \
  --setting ignore_bot_messages=true \
  --setting workspace_id=telegram
```

Setting notes:

- `token_env`: required env var containing the bot token
- `chat_id`: single numeric Telegram chat id
- `chat_ids`: optional list of numeric Telegram chat ids, or a comma-separated string of ids
- `pairing_mode`: `off`, `pending`, or `auto`; default `off`
- `pairing_users`: optional list of senders who may admit/pair new rooms; accepts Telegram user ids or usernames
- `allowed_users`: optional list of senders who may interact in approved rooms; empty means any sender in an approved room
- `banned_users`: optional list of senders who are always denied; overrides `allowed_users`
- `respond_mode`: `all`, `mentions`, `replies`, or `mentions_or_replies`; default `all`
- `session_scope`: `user`, `thread`, or `room`; default `user`
- `poll_timeout_secs`: long-poll timeout, default `30`, maximum `50`
- `poll_interval_ms`: delay between empty polls, default `250`
- `task_timeout_ms`: optional Turin task wait timeout for this channel; `0` or omitted means wait indefinitely
- `stream_mode`: `off`, `typing`, `draft`, or `block`; default `off`
- `stream_thinking`: optional boolean, default `false`; when enabled, `draft`/`block` previews can include streamed model thinking if the provider emits thinking deltas
- `persist_thinking`: optional boolean, default `false`; when enabled, final Telegram replies keep the thinking block above the answer
- `start_from_latest`: skip old queued updates at startup
- `ignore_bot_messages`: ignore bot-authored inbound messages
- `workspace_id`: optional routing namespace label
- `base_url`: optional override for Telegram-compatible endpoints or tests

When `pairing_mode` is `pending`, inspect and manage discovered rooms with:

```bash
turin daemon channel access telegram-ops
turin daemon channel approve telegram-ops --workspace-id telegram --room-id -1001234567890 --thread-id -1001234567890
turin daemon channel reject telegram-ops --workspace-id telegram --room-id -1001234567890 --thread-id -1001234567890
turin daemon channel revoke telegram-ops --workspace-id telegram --room-id -1001234567890 --thread-id -1001234567890
```

`approve` moves a room from pending to approved. `reject` clears a pending room without approving it. `revoke` removes a previously approved room.

## 7. Verify Runtime Status

Check whether the runtime reached `running`:

```bash
turin daemon channel status telegram-ops
```

You can also inspect the daemon-wide runtime view:

```bash
turin daemon status --json
```

If startup fails, check:

```bash
turin daemon channel issues telegram-ops
turin daemon channel status telegram-ops --json
```

The normalized `last_error_code` is especially useful for fast diagnosis.

## 8. Equivalent `channel.toml`

The daemon stores channel settings under `channels/<id>/channel.toml`.

Example:

```toml
enabled = true
kind = "telegram"
agent_id = "default"
idle_ttl_secs = 600
token_env = "TELEGRAM_BOT_TOKEN"
pairing_mode = "auto"
pairing_users = ["498502840"]
poll_timeout_secs = 10
poll_interval_ms = 250
respond_mode = "mentions_or_replies"
stream_mode = "block"
stream_thinking = false
persist_thinking = false
start_from_latest = true
ignore_bot_messages = true
workspace_id = "telegram"
```

You can manage this file directly through the filesystem or via `turin daemon channel ...` commands.

If you prefer explicit room allowlisting instead of pairing, replace the pairing settings with:

```toml
chat_ids = [-1001234567890, -100987654321, 498502840]
```

## 9. Outbound Reply And Formatting Behavior

Turin now defaults Telegram responses to replying to the inbound Telegram message when the inbound event includes `telegram_message_id`.

That means normal request/response turns in Telegram show up as native threaded replies without extra harness work.

If you emit a structured outbound payload, you can also control Telegram-specific behavior through outbound metadata:

- `telegram_reply_to_message_id`: explicit numeric reply target, or `null` to suppress the default reply-to behavior
- `telegram_disable_web_page_preview`: defaults to `true`
- `telegram_disable_notification`: defaults to `false`
- `telegram_format`: `plain`/`text` to force plain text, or `html` to force Telegram HTML parse mode
- `telegram_parse_mode`: currently supports `html`

Formatting notes:

- Turin now defaults Telegram text rendering to Telegram HTML parse mode.
- Common Markdown-style agent output such as headings, bold/italic, inline code, links, lists, block quotes, and fenced code blocks is rendered into Telegram-safe HTML automatically.
- Markdown tables are rendered as aligned monospaced text inside Telegram `<pre>` blocks.
- `telegram_format = "plain"` disables that rendering and sends raw text instead.
- Attachments are still rendered as text lines; media upload is not part of this adapter yet.

Streaming notes:

- `stream_mode = "off"` keeps the old behavior: no typing indicator and only a final reply.
- `stream_mode = "typing"` sends Telegram `typing` actions while Turin is working, then sends the final reply.
- `stream_mode = "draft"` streams a partial preview while the response is being generated. Turin prefers Telegram draft streaming in private chats and falls back to a bot-authored placeholder message plus edits when drafts are unavailable.
- `stream_mode = "block"` is like `draft`, but updates less frequently and favors chunkier preview steps over every small delta.
- `stream_thinking = true` is an additional opt-in. When paired with `draft` or `block`, preview messages can include streamed model thinking before or alongside the partial answer. It has no visible effect with `off` or `typing`.
- `persist_thinking = true` keeps the model thinking in the final Telegram reply, rendered as a separate preformatted block above the answer.
- Thinking previews only appear when the selected model/provider actually emits thinking deltas.
- Final replies still use the normal Telegram HTML renderer even when preview streaming is enabled.

## 10. Validate With The Smoke Script

For a quick live validation against the real Telegram Bot API:

```bash
scripts/live_telegram_channel_smoke.sh \
  --chat-id "$TELEGRAM_CHAT_ID" \
  --token-env-name TELEGRAM_BOT_TOKEN
```

This brings up a temporary Turin workspace, creates a `kind=telegram` channel, and checks that the runtime reaches `running`.

## Troubleshooting

### `telegram_auth_missing_token`

Meaning:
- the env var named by `token_env` is not set in the daemon environment

Fix:
- export the token before starting the daemon
- or restart the daemon from an environment where the token exists

### Polling/webhook conflict

Meaning:
- the bot still has an active webhook, so Telegram rejects `getUpdates`

Fix:

```bash
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/deleteWebhook?drop_pending_updates=true"
```

Then restart or re-enable the channel.

### Intermittent Telegram API failures or rate limits

Meaning:
- Telegram returned a transient send/poll error such as rate limiting or a temporary upstream failure

What Turin does:
- the Telegram adapter retries the individual API call with bounded backoff
- if polling still cannot recover, the runtime keeps backing off instead of immediately crashing on the first transient error

What to check:
- whether the bot is being rate limited by repeated test traffic
- whether another process is also polling the same bot token
- whether Telegram still reports an active webhook

### No updates arrive from a group

Likely causes:

- privacy mode is still enabled
- the bot was added but no new message was sent afterward
- the channel is pointed at the wrong `chat_id`
- pairing is `off` and the group is not in `chat_ids`
- the sender is not allowed by `pairing_users` or `allowed_users`
- the sender is explicitly denied by `banned_users`
- `respond_mode` is set to `mentions`, `replies`, or `mentions_or_replies`, and the test message did not match that policy

### No updates arrive from a channel

Likely causes:

- the bot is not an admin in the channel
- you are looking at `message.chat.id` instead of `channel_post.chat.id`
- the channel is pointed at the wrong `chat_id`

### Forum topics do not stay separated

Check whether the incoming Telegram payload actually contains `message_thread_id`.

Turin routes per topic only when Telegram includes that field. The base `chat_id` is still shared across the supergroup.
