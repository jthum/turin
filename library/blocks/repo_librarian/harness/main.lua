local function read_required(path)
  local text, err = try(fs.read, path)
  if not text then
    error("required repo librarian file missing: " .. path .. ": " .. tostring(err))
  end
  return text
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write repo librarian artifact " .. path .. ": " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local soul = read_required("SOUL.md")
  local agents = read_required("AGENTS.md")
  local architecture = read_required("ARCHITECTURE.md")
  local conventions = read_required("CONVENTIONS.md")
  local contract = table.concat({
    "# SOUL.md",
    soul,
    "",
    "# AGENTS.md",
    agents,
    "",
    "# ARCHITECTURE.md",
    architecture,
    "",
    "# CONVENTIONS.md",
    conventions,
  }, "\n")

  session.incr("repo_librarian.run_count")
  session.set("repo_librarian.last_prompt", prompt)

  runtime.db.with(".turin/runtime/harness.db", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS repo_librarian_runs (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])
    db:exec(
      "INSERT INTO repo_librarian_runs(prompt, created_at) VALUES (?, ?)",
      { prompt, time.now_utc() }
    )
  end)

  write_required(".turin/runtime/repo-librarian/contract.md", contract)
  write_required(".turin/runtime/repo-librarian/last-request.txt", prompt)

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Repository contract:",
    contract,
    "",
    "You are the repo librarian. Route work and shape advice according to the repository's contracts, architecture, and conventions.",
  }, "\n")

  return ALLOW
end
