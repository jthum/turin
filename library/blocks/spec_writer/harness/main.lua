local function read_required(path)
  local text, err = try(fs.read, path)
  if not text then
    error("required spec writer file missing: " .. path .. ": " .. tostring(err))
  end
  return text
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write spec writer artifact " .. path .. ": " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local idea = read_required("IDEA.md")
  local acceptance = read_required("ACCEPTANCE.md")
  local context = read_required("CONTEXT.md")
  local contract = table.concat({
    "# IDEA.md",
    idea,
    "",
    "# ACCEPTANCE.md",
    acceptance,
    "",
    "# CONTEXT.md",
    context,
  }, "\n")

  session.incr("spec_writer.run_count")
  session.set("spec_writer.last_prompt", prompt)

  runtime.db.with(".turin/runtime/harness.db", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS spec_writer_runs (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])
    db:exec(
      "INSERT INTO spec_writer_runs(prompt, created_at) VALUES (?, ?)",
      { prompt, time.now_utc() }
    )
  end)

  write_required(".turin/runtime/spec-writer/contract.md", contract)
  write_required(".turin/runtime/spec-writer/last-request.txt", prompt)

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Specification contract:",
    contract,
    "",
    "You are the spec writer. Turn rough ideas into practical implementation specs with scope, constraints, and acceptance criteria.",
  }, "\n")

  return ALLOW
end
