local function read_optional(path)
  local text = fs.read(path)
  return text or ""
end

local function write_required(path, content)
  local ok, err = fs.write(path, content)
  if not ok then
    error("failed to write planner runtime artifact: " .. tostring(err))
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
    "You are the planner agent. Produce concrete plans, sequencing, and next steps.",
  }, "\n")

  session.set("personal_assistant.role", "planner")
  write_required(".turin/runtime/personal-assistant/planner-last-request.txt", prompt)
  return ALLOW
end
