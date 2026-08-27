local function read_required(path)
  local text, err = try(fs.read, path)
  if not text then
    error("required task planner file missing: " .. path .. ": " .. tostring(err))
  end
  return text
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write task planner artifact " .. path .. ": " .. tostring(err))
  end
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local planning_style = read_required("PLANNING_STYLE.md")
  local delivery_constraints = read_required("DELIVERY_CONSTRAINTS.md")
  local context = table.concat({
    "# PLANNING_STYLE.md",
    planning_style,
    "",
    "# DELIVERY_CONSTRAINTS.md",
    delivery_constraints,
  }, "\n")

  session.incr("task_planner.run_count")
  session.set("task_planner.last_prompt", prompt)

  runtime.db.with(".turin/runtime/harness.db", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS task_planner_runs (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])
    db:exec(
      "INSERT INTO task_planner_runs(prompt, created_at) VALUES (?, ?)",
      { prompt, time.now_utc() }
    )
  end)

  write_required(".turin/runtime/task-planner/context.md", context)
  write_required(".turin/runtime/task-planner/last-request.txt", prompt)

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Planning contract:",
    context,
    "",
    "You are the task planner. Produce concrete, sequenced next actions with dependencies and validation steps.",
  }, "\n")

  return ALLOW
end
