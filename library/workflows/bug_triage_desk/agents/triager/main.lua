local function read_optional(path)
  local text = try(fs.read, path)
  return text or ""
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write triager artifact: " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local severity_policy = read_optional("SEVERITY_POLICY.md")
  local ownership = read_optional("OWNERSHIP.md")
  local prompt = tostring(ctx.prompt or "")

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Severity policy:",
    severity_policy,
    "",
    "Ownership map:",
    ownership,
    "",
    "You are the triager. Classify severity, likely owner, and immediate next checks.",
  }, "\n")

  session.set("bug_triage.role", "triager")
  write_required(".turin/runtime/bug-triage/triager-last-request.txt", prompt)
  return ALLOW
end
