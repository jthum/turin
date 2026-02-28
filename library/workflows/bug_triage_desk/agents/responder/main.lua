local function read_optional(path)
  local text = fs.read(path)
  return text or ""
end

local function write_required(path, content)
  local ok, err = fs.write(path, content)
  if not ok then
    error("failed to write responder artifact: " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local known_issues = read_optional("KNOWN_ISSUES.md")
  local runbook = read_optional("RUNBOOK.md")
  local prompt = tostring(ctx.prompt or "")

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Known issues:",
    known_issues,
    "",
    "Runbook:",
    runbook,
    "",
    "You are the responder. Draft an operator-facing response and next-action checklist.",
  }, "\n")

  session.set("bug_triage.role", "responder")
  write_required(".turin/runtime/bug-triage/responder-last-request.txt", prompt)
  return ALLOW
end
