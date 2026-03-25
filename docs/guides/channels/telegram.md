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

## 2. Decide Which Telegram Surface You Want

Turin’s `chat_id` is a numeric Telegram chat identifier. The setup differs slightly by chat type.

### Direct bot chat

Send a message to the bot directly so Telegram creates the conversation.

### Group or supergroup

Add the bot to the group and send a test message.

If you want the bot to receive ordinary group messages, disable privacy mode in BotFather:

1. Run `/setprivacy`.
2. Select the bot.
3. Choose `Disable`.

If privacy mode stays enabled, Telegram may only deliver commands, replies, and mentions instead of normal text traffic.

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

If you do not see the chat you expect:

- make sure you sent a fresh test message after adding the bot
- make sure privacy mode is disabled for normal group traffic
- make sure the bot is an admin if you are targeting a channel
- make sure no webhook is still active

## 5. Create The Turin Channel

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
  --setting chat_id=-1001234567890 \
  --setting poll_timeout_secs=10 \
  --setting poll_interval_ms=250 \
  --setting start_from_latest=true \
  --setting ignore_bot_messages=true \
  --setting workspace_id=telegram
```

Setting notes:

- `token_env`: required env var containing the bot token
- `chat_id`: required numeric Telegram chat id
- `poll_timeout_secs`: long-poll timeout, default `30`, maximum `50`
- `poll_interval_ms`: delay between empty polls, default `250`
- `start_from_latest`: skip old queued updates at startup
- `ignore_bot_messages`: ignore bot-authored inbound messages
- `workspace_id`: optional routing namespace label
- `base_url`: optional override for Telegram-compatible endpoints or tests

## 6. Verify Runtime Status

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

## 7. Equivalent `channel.toml`

The daemon stores channel settings under `channels/<id>/channel.toml`.

Example:

```toml
enabled = true
kind = "telegram"
agent_id = "default"
idle_ttl_secs = 600
token_env = "TELEGRAM_BOT_TOKEN"
chat_id = -1001234567890
poll_timeout_secs = 10
poll_interval_ms = 250
start_from_latest = true
ignore_bot_messages = true
workspace_id = "telegram"
```

You can manage this file directly through the filesystem or via `turin daemon channel ...` commands.

## 8. Outbound Reply And Formatting Behavior

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
- `telegram_format = "plain"` disables that rendering and sends raw text instead.
- Attachments are still rendered as text lines; media upload is not part of this adapter yet.

## 9. Validate With The Smoke Script

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

### No updates arrive from a channel

Likely causes:

- the bot is not an admin in the channel
- you are looking at `message.chat.id` instead of `channel_post.chat.id`
- the channel is pointed at the wrong `chat_id`

### Forum topics do not stay separated

Check whether the incoming Telegram payload actually contains `message_thread_id`.

Turin routes per topic only when Telegram includes that field. The base `chat_id` is still shared across the supergroup.
