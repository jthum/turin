local function read_optional(path)
  local text = try(fs.read, path)
  return text or ""
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write readiness reviewer artifact: " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local goals = read_optional("RELEASE_GOALS.md")
  local checklist = read_optional("CHECKLIST.md")
  local prompt = tostring(ctx.prompt or "")

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Release goals:",
    goals,
    "",
    "Checklist:",
    checklist,
    "",
    "You are the readiness reviewer. Focus on blockers, risks, and missing validation.",
  }, "\n")

  session.set("release_manager.role", "readiness_reviewer")
  write_required(".turin/runtime/release-manager/readiness-reviewer-last-request.txt", prompt)
  return ALLOW
end
