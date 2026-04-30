local function read_optional(path)
  local text = try(fs.read, path)
  return text or ""
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write docs reviewer artifact: " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local public_surface = read_optional("PUBLIC_SURFACE.md")
  local targets = read_optional("DOCS_TARGETS.md")
  local prompt = tostring(ctx.prompt or "")

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Public surface:",
    public_surface,
    "",
    "Docs targets:",
    targets,
    "",
    "You are the docs reviewer. Identify drift, stale claims, and exact docs that need updates.",
  }, "\n")

  session.set("docs_team_assistant.role", "docs_reviewer")
  write_required(".turin/runtime/docs-team/docs-reviewer-last-request.txt", prompt)
  return ALLOW
end
