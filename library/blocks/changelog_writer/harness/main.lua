local function read_required(path)
  local text, err = try(fs.read, path)
  if not text then
    error("required changelog writer file missing: " .. path .. ": " .. tostring(err))
  end
  return text
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write changelog writer artifact " .. path .. ": " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local release_scope = read_required("RELEASE_SCOPE.md")
  local merged_changes = read_required("MERGED_CHANGES.md")
  local writing_style = read_required("WRITING_STYLE.md")
  local contract = table.concat({
    "# RELEASE_SCOPE.md",
    release_scope,
    "",
    "# MERGED_CHANGES.md",
    merged_changes,
    "",
    "# WRITING_STYLE.md",
    writing_style,
  }, "\n")

  session.incr("changelog_writer.run_count")
  session.set("changelog_writer.last_prompt", prompt)

  runtime.db.with(".turin/runtime/harness.db", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS changelog_writer_runs (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])
    db:exec(
      "INSERT INTO changelog_writer_runs(prompt, created_at) VALUES (?, ?)",
      { prompt, time.now_utc() }
    )
  end)

  write_required(".turin/runtime/changelog-writer/contract.md", contract)
  write_required(".turin/runtime/changelog-writer/last-request.txt", prompt)

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Changelog writing contract:",
    contract,
    "",
    "You are the changelog writer. Produce concise, operator-facing release notes grounded in the merged changes.",
  }, "\n")

  return ALLOW
end
