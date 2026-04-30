local function read_optional(path)
  local text = try(fs.read, path)
  return text or ""
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write draft writer artifact: " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local drift = read_optional("DRIFT_NOTES.md")
  local style = read_optional("STYLE_NOTES.md")
  local prompt = tostring(ctx.prompt or "")

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Drift notes:",
    drift,
    "",
    "Style notes:",
    style,
    "",
    "You are the draft writer. Produce concise documentation update text grounded in the review findings.",
  }, "\n")

  session.set("docs_team_assistant.role", "draft_writer")
  write_required(".turin/runtime/docs-team/draft-writer-last-request.txt", prompt)
  return ALLOW
end
