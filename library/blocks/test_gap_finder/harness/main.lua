local function read_required(path)
  local text, err = try(fs.read, path)
  if not text then
    error("required test gap file missing: " .. path .. ": " .. tostring(err))
  end
  return text
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write test gap artifact " .. path .. ": " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local change_summary = read_required("CHANGE_SUMMARY.md")
  local testing_policy = read_required("TESTING_POLICY.md")
  local risk_areas = read_required("RISK_AREAS.md")
  local contract = table.concat({
    "# CHANGE_SUMMARY.md",
    change_summary,
    "",
    "# TESTING_POLICY.md",
    testing_policy,
    "",
    "# RISK_AREAS.md",
    risk_areas,
  }, "\n")

  session.incr("test_gap_finder.run_count")
  session.set("test_gap_finder.last_prompt", prompt)

  runtime.db.with("state", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS test_gap_runs (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])
    db:exec(
      "INSERT INTO test_gap_runs(prompt, created_at) VALUES (?, ?)",
      { prompt, time.now_utc() }
    )
  end)

  write_required(".turin/runtime/test-gap-finder/contract.md", contract)
  write_required(".turin/runtime/test-gap-finder/last-request.txt", prompt)

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Test-gap contract:",
    contract,
    "",
    "You are the test gap finder. Identify likely missing tests, risky untested paths, and what should be verified before shipping.",
  }, "\n")

  return ALLOW
end
