local function read_required(path)
  local text, err = try(fs.read, path)
  if not text then
    error("required coding harness file missing: " .. path .. ": " .. tostring(err))
  end
  return text
end

local function read_optional(path)
  local text = try(fs.read, path)
  return text or nil
end

local function write_required(path, content)
  local ok, err = try(fs.write, path, content)
  if not ok then
    error("failed to write coding harness artifact " .. path .. ": " .. tostring(err))
  end
end

local function build_context(prompt)
  local spec = read_required("SPEC.md")
  local tasks = read_required("TASKS.md")
  local constraints = read_required("CONSTRAINTS.md")
  local notes = read_optional("NOTES.md")

  local sections = {
    "# SPEC.md",
    spec,
    "",
    "# TASKS.md",
    tasks,
    "",
    "# CONSTRAINTS.md",
    constraints,
  }

  if notes and notes ~= "" then
    table.insert(sections, "")
    table.insert(sections, "# NOTES.md")
    table.insert(sections, notes)
  end

  local context = table.concat(sections, "\n")
  local planner_prompt = table.concat({
    "You are preparing a coding execution plan.",
    "Focus on sequencing, scope, and likely files to touch.",
    "",
    "User request:",
    prompt,
    "",
    "Workspace context:",
    context,
  }, "\n")

  return {
    context = context,
    planner_prompt = planner_prompt,
    reviewer_prompt_prefix = table.concat({
      "Review the proposed coding plan for regressions, missing tests, and operational risk.",
      "Focus on what could go wrong and what should be verified before changes land.",
      "",
      "User request:",
      prompt,
    }, "\n"),
  }
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local built = build_context(prompt)
  local planner = runtime.agent("planner")
  local reviewer = runtime.agent("reviewer")

  local plan = planner:ask(built.planner_prompt)
  if plan == nil or plan == "" then
    error("planner returned empty output")
  end

  local review = reviewer:ask(table.concat({
    built.reviewer_prompt_prefix,
    "",
    "Proposed plan:",
    plan,
  }, "\n"))
  if review == nil or review == "" then
    error("reviewer returned empty output")
  end

  local brief = table.concat({
    "# Coding Harness Brief",
    "",
    "Prompt: " .. prompt,
    "Generated at: " .. time.now_utc(),
    "",
    "## Plan",
    plan,
    "",
    "## Review",
    review,
  }, "\n")

  session.incr("coding_harness.run_count")
  session.set("coding_harness.last_prompt", prompt)
  session.set("coding_harness.plan_bytes", tostring(#plan))
  session.set("coding_harness.review_bytes", tostring(#review))

  runtime.db.with(".turin/runtime/harness.db", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS coding_harness_runs (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        plan TEXT NOT NULL,
        review TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])

    db:exec(
      "INSERT INTO coding_harness_runs(prompt, plan, review, created_at) VALUES (?, ?, ?, ?)",
      { prompt, plan, review, time.now_utc() }
    )
  end)

  write_required(".turin/runtime/coding-harness/context.md", built.context)
  write_required(".turin/runtime/coding-harness/plan.md", plan)
  write_required(".turin/runtime/coding-harness/review.md", review)
  write_required(".turin/runtime/coding-harness/brief.md", brief)
  write_required(".turin/runtime/coding-harness/last-prompt.txt", prompt)

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Coding workspace context:",
    built.context,
    "",
    "Execution plan:",
    plan,
    "",
    "Review checklist:",
    review,
  }, "\n")

  return ALLOW
end
