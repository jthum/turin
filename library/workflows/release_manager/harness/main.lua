local function read_required(path)
  local text, err = fs.read(path)
  if not text then
    error("required release manager file missing: " .. path .. ": " .. tostring(err))
  end
  return text
end

local function read_optional(path)
  local text = fs.read(path)
  return text or nil
end

local function write_required(path, content)
  local ok, err = fs.write(path, content)
  if not ok then
    error("failed to write release manager artifact " .. path .. ": " .. tostring(err))
  end
end

local function build_context(prompt)
  local goals = read_required("RELEASE_GOALS.md")
  local notes = read_required("CHANGELOG_NOTES.md")
  local issues = read_required("OPEN_ISSUES.md")
  local checklist = read_required("CHECKLIST.md")
  local constraints = read_optional("CONSTRAINTS.md")

  local sections = {
    "# RELEASE_GOALS.md",
    goals,
    "",
    "# CHANGELOG_NOTES.md",
    notes,
    "",
    "# OPEN_ISSUES.md",
    issues,
    "",
    "# CHECKLIST.md",
    checklist,
  }

  if constraints and constraints ~= "" then
    table.insert(sections, "")
    table.insert(sections, "# CONSTRAINTS.md")
    table.insert(sections, constraints)
  end

  local context = table.concat(sections, "\n")
  local review_prompt = table.concat({
    "Assess release readiness for the following release request.",
    "Focus on risks, blockers, missing validation, and whether shipping now looks reasonable.",
    "",
    "Release request:",
    prompt,
    "",
    "Release context:",
    context,
  }, "\n")

  return {
    context = context,
    review_prompt = review_prompt,
    changelog_prompt_prefix = table.concat({
      "Draft concise release notes for the following release request.",
      "Use the checked-in notes and the readiness review as grounding.",
      "",
      "Release request:",
      prompt,
    }, "\n"),
  }
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local built = build_context(prompt)
  local reviewer = runtime.agent("readiness_reviewer")
  local changelog_writer = runtime.agent("changelog_writer")

  local readiness = reviewer:ask(built.review_prompt)
  if readiness == nil or readiness == "" then
    error("readiness reviewer returned empty output")
  end

  local changelog = changelog_writer:ask(table.concat({
    built.changelog_prompt_prefix,
    "",
    "Readiness review:",
    readiness,
  }, "\n"))
  if changelog == nil or changelog == "" then
    error("changelog writer returned empty output")
  end

  local brief = table.concat({
    "# Release Manager Brief",
    "",
    "Release request: " .. prompt,
    "Generated at: " .. time.now_utc(),
    "",
    "## Readiness Review",
    readiness,
    "",
    "## Draft Release Notes",
    changelog,
  }, "\n")

  session.incr("release_manager.run_count")
  session.set("release_manager.last_prompt", prompt)
  session.set("release_manager.readiness_bytes", tostring(#readiness))
  session.set("release_manager.changelog_bytes", tostring(#changelog))

  runtime.db.with("state", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS release_manager_runs (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        readiness TEXT NOT NULL,
        changelog TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])
    db:exec(
      "INSERT INTO release_manager_runs(prompt, readiness, changelog, created_at) VALUES (?, ?, ?, ?)",
      { prompt, readiness, changelog, time.now_utc() }
    )
  end)

  write_required(".turin/runtime/release-manager/context.md", built.context)
  write_required(".turin/runtime/release-manager/readiness.md", readiness)
  write_required(".turin/runtime/release-manager/changelog.md", changelog)
  write_required(".turin/runtime/release-manager/brief.md", brief)
  write_required(".turin/runtime/release-manager/last-prompt.txt", prompt)

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Release context:",
    built.context,
    "",
    "Readiness review:",
    readiness,
    "",
    "Draft release notes:",
    changelog,
  }, "\n")

  return ALLOW
end
