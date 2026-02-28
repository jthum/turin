local function read_optional(path)
  local text = fs.read(path)
  return text or ""
end

local function write_required(path, content)
  local ok, err = fs.write(path, content)
  if not ok then
    error("failed to write changelog writer artifact: " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local notes = read_optional("CHANGELOG_NOTES.md")
  local issues = read_optional("OPEN_ISSUES.md")
  local prompt = tostring(ctx.prompt or "")

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Changelog notes:",
    notes,
    "",
    "Open issues:",
    issues,
    "",
    "You are the changelog writer. Produce concise release notes grounded in the supplied context.",
  }, "\n")

  session.set("release_manager.role", "changelog_writer")
  write_required(".turin/runtime/release-manager/changelog-writer-last-request.txt", prompt)
  return ALLOW
end
