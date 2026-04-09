# Multimodal Phase 1

Turin Phase 1 multimodal support adds image and generic file references to task input without turning core into a media-processing runtime.

## Scope

Current scope:

- image references
- generic file references
- durable local storage for inbound media
- session-history persistence and restore for those refs
- channel ingress and outbound delivery on adapters that implement attachments

Not in scope yet:

- audio
- video
- OCR, transcription, or other media parsing inside Turin core
- attachment-specific policy beyond the existing text cap

## Storage Model

Turin does not store attachment bytes in SQLite blobs by default.

Instead:

- message content metadata is persisted in the normal session transcript JSON
- attachment bytes are copied or downloaded into `.turin/data/media/`
- persisted message history stores the managed `local_path` alongside the original `url` when available

That keeps the database small while making resumed sessions and provider requests durable.

## Harness Behavior

`ctx.messages` can now include `image` and `file` parts inside user messages.

Current part shapes are:

```lua
{ type = "text", text = "Inspect this" }
{ type = "image", name = "diagram.png", content_type = "image/png", url = "...", local_path = "...", detail = "high" }
{ type = "file", name = "spec.pdf", content_type = "application/pdf", url = "...", local_path = "..." }
```

`ctx.prompt` remains text-only:

- it is derived from text parts of the latest user message
- image-only or file-only messages leave `ctx.prompt` unset
- assigning `ctx.prompt` replaces only the text parts of the latest user message and preserves non-text attachments

If a harness needs to reason about attachments, inspect `ctx.messages` directly.

## Provider Behavior

Phase 1 keeps provider support pragmatic:

- OpenAI-compatible drivers map image refs into native image content parts
- Anthropic-compatible drivers map image refs into native base64 image blocks
- generic file refs are preserved in Turin history and request state, but current providers receive them as text fallback summaries rather than native document parts

That means images are first-class in supported providers now, while generic files already survive end to end and can be handled by later provider-native upgrades.

## Channel Support

End-to-end attachment support currently depends on the adapter:

- Discord: inbound attachments are forwarded into Turin task content, and outbound local file attachments are supported
- FS channel: inbound and outbound attachment refs are supported
- Rocket.Chat: inbound attachment refs exist, but outbound file upload is not implemented yet
- Telegram: inbound image/document/video/audio/voice attachments are downloaded into managed local storage, and outbound image/document attachments are uploaded through the Telegram Bot API
- WhatsApp: inbound media attachments are downloaded into managed local storage, and outbound local file attachments are uploaded through the linked-device session

Current limitations:

- Telegram outbound uploads currently map images to `sendPhoto` and everything else to `sendDocument`
- WhatsApp outbound uploads currently require `local_path`; plain remote URLs are not uploaded directly
- Rocket.Chat still has no outbound file-upload path

## Outbound Attachments

Outbound channel attachments can come from either:

- the explicit structured envelope
- assistant content parts (`text`, `image`, `file`) returned by Turin itself

The structured envelope still takes precedence when you need channel-specific control:

```json
{
  "_turin_channel_outbound": true,
  "content": "Build summary",
  "attachments": [
    {
      "name": "report.txt",
      "local_path": "/abs/path/report.txt",
      "content_type": "text/plain"
    }
  ]
}
```

If no structured envelope is present, the channel runner now automatically maps assistant content parts into outbound blocks and attachments on adapters that implement outbound file delivery.

This is distinct from provider-side multimodal input. Phase 1 lets channels send attachments into Turin task content on supported adapters and can also map assistant attachment parts back into channel uploads where the adapter supports them.
