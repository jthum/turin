local function read_required(path)
  local text, err = try(fs.read, path)
  if not text then
    error("required release readiness file missing: " .. path .. ": " .. tostring(err))
  end
  return text
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write release readiness artifact " .. path .. ": " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local checklist = read_required("CHECKLIST.md")
  local risk_register = read_required("RISK_REGISTER.md")
  local release_notes = read_required("RELEASE_NOTES_CONTEXT.md")
  local contract = table.concat({
    "# CHECKLIST.md",
    checklist,
    "",
    "# RISK_REGISTER.md",
    risk_register,
    "",
    "# RELEASE_NOTES_CONTEXT.md",
    release_notes,
  }, "\n")

  session.incr("release_readiness_checker.run_count")
  session.set("release_readiness_checker.last_prompt", prompt)

  runtime.db.with(".turin/runtime/harness.db", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS release_readiness_runs (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])
    db:exec(
      "INSERT INTO release_readiness_runs(prompt, created_at) VALUES (?, ?)",
      { prompt, time.now_utc() }
    )
  end)

  write_required(".turin/runtime/release-readiness/contract.md", contract)
  write_required(".turin/runtime/release-readiness/last-request.txt", prompt)

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Release readiness contract:",
    contract,
    "",
    "You are the release readiness checker. Call out blockers, missing validation, and whether shipping now looks justified.",
  }, "\n")

  return ALLOW
end
