local function read_optional(path)
  local text = try(fs.read, path)
  return text or ""
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write reviewer runtime artifact: " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local soul = read_optional("SOUL.md")
  local profile = read_optional("PROFILE.md")
  local prompt = tostring(ctx.prompt or "")

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Shared contract:",
    soul,
    "",
    "User profile:",
    profile,
    "",
    "You are the reviewer agent. Focus on regressions, risk, and missing checks.",
  }, "\n")

  session.set("personal_assistant.role", "reviewer")
  write_required(".turin/runtime/personal-assistant/reviewer-last-request.txt", prompt)
  return ALLOW
end
