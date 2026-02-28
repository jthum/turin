local function read_required(path)
  local text, err = fs.read(path)
  if not text then
    error("required contract file missing: " .. path .. ": " .. tostring(err))
  end
  return text
end

function on_turn_prepare(ctx)
  local soul = read_required("SOUL.md")
  local agents = read_required("AGENTS.md")
  local contract = table.concat({
    "# SOUL.md",
    soul,
    "",
    "# AGENTS.md",
    agents,
  }, "\n")

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Workspace contract:",
    contract,
  }, "\n")

  session.set("openclaw.contract_loaded_at", time.now_utc())
  session.set("openclaw.contract_bytes", tostring(#contract))
  session.set("openclaw.last_prompt", tostring(ctx.prompt or ""))

  fs.write(".turin/runtime/openclaw-contract.md", contract)
  fs.write(".turin/runtime/openclaw-last-prompt.txt", tostring(ctx.prompt or ""))
  return ALLOW
end
