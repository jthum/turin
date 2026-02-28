local function read_required(path)
  local text, err = fs.read(path)
  if not text then
    error("required code reviewer file missing: " .. path .. ": " .. tostring(err))
  end
  return text
end

local function write_required(path, content)
  local ok, err = fs.write(path, content)
  if not ok then
    error("failed to write code reviewer artifact " .. path .. ": " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local review_style = read_required("REVIEW_STYLE.md")
  local risk_areas = read_required("RISK_AREAS.md")
  local context = table.concat({
    "# REVIEW_STYLE.md",
    review_style,
    "",
    "# RISK_AREAS.md",
    risk_areas,
  }, "\n")

  session.incr("code_reviewer.run_count")
  session.set("code_reviewer.last_prompt", prompt)

  runtime.db.with("state", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS code_review_runs (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])
    db:exec(
      "INSERT INTO code_review_runs(prompt, created_at) VALUES (?, ?)",
      { prompt, time.now_utc() }
    )
  end)

  write_required(".turin/runtime/code-review/context.md", context)
  write_required(".turin/runtime/code-review/last-request.txt", prompt)

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Review contract:",
    context,
    "",
    "You are the code reviewer. Focus on regressions, missing tests, and risky assumptions.",
  }, "\n")

  return ALLOW
end
