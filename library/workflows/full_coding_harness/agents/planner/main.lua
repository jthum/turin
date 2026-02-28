local function read_optional(path)
  local text = fs.read(path)
  return text or ""
end

local function write_required(path, content)
  local ok, err = fs.write(path, content)
  if not ok then
    error("failed to write planner artifact: " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local spec = read_optional("SPEC.md")
  local constraints = read_optional("CONSTRAINTS.md")
  local prompt = tostring(ctx.prompt or "")

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Shared spec context:",
    spec,
    "",
    "Constraints:",
    constraints,
    "",
    "You are the planner for the coding harness. Produce concrete steps, likely files, and validation tasks.",
  }, "\n")

  session.set("coding_harness.role", "planner")
  write_required(".turin/runtime/coding-harness/planner-last-request.txt", prompt)
  return ALLOW
end
