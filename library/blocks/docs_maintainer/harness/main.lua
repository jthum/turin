local function read_required(path)
  local text, err = try(fs.read, path)
  if not text then
    error("required docs maintainer file missing: " .. path .. ": " .. tostring(err))
  end
  return text
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write docs maintainer artifact " .. path .. ": " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local public_surface = read_required("PUBLIC_SURFACE.md")
  local docs_policy = read_required("DOCS_POLICY.md")
  local drift_signals = read_required("DRIFT_SIGNALS.md")
  local contract = table.concat({
    "# PUBLIC_SURFACE.md",
    public_surface,
    "",
    "# DOCS_POLICY.md",
    docs_policy,
    "",
    "# DRIFT_SIGNALS.md",
    drift_signals,
  }, "\n")

  session.incr("docs_maintainer.run_count")
  session.set("docs_maintainer.last_prompt", prompt)

  runtime.db.with(".turin/runtime/harness.db", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS docs_maintainer_runs (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])
    db:exec(
      "INSERT INTO docs_maintainer_runs(prompt, created_at) VALUES (?, ?)",
      { prompt, time.now_utc() }
    )
  end)

  write_required(".turin/runtime/docs-maintainer/contract.md", contract)
  write_required(".turin/runtime/docs-maintainer/last-request.txt", prompt)

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Documentation maintenance contract:",
    contract,
    "",
    "You are the docs maintainer. Identify documentation drift and propose updates grounded in the current public surface.",
  }, "\n")

  return ALLOW
end
