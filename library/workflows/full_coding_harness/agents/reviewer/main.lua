local function read_optional(path)
  local text = try(fs.read, path)
  return text or ""
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write reviewer artifact: " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local tasks = read_optional("TASKS.md")
  local constraints = read_optional("CONSTRAINTS.md")
  local prompt = tostring(ctx.prompt or "")

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Shared task context:",
    tasks,
    "",
    "Constraints:",
    constraints,
    "",
    "You are the reviewer for the coding harness. Focus on regressions, test gaps, and unsafe assumptions.",
  }, "\n")

  session.set("coding_harness.role", "reviewer")
  write_required(".turin/runtime/coding-harness/reviewer-last-request.txt", prompt)
  return ALLOW
end
