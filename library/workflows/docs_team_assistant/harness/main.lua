local function read_required(path)
  local text, err = try(fs.read, path)
  if not text then
    error("required docs team file missing: " .. path .. ": " .. tostring(err))
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
    error("failed to write docs team artifact " .. path .. ": " .. tostring(err))
  end
end

local function build_context(prompt)
  local public_surface = read_required("PUBLIC_SURFACE.md")
  local docs_targets = read_required("DOCS_TARGETS.md")
  local drift = read_required("DRIFT_NOTES.md")
  local style = read_optional("STYLE_NOTES.md")

  local sections = {
    "# PUBLIC_SURFACE.md",
    public_surface,
    "",
    "# DOCS_TARGETS.md",
    docs_targets,
    "",
    "# DRIFT_NOTES.md",
    drift,
  }

  if style and style ~= "" then
    table.insert(sections, "")
    table.insert(sections, "# STYLE_NOTES.md")
    table.insert(sections, style)
  end

  local context = table.concat(sections, "\n")
  local review_prompt = table.concat({
    "Review the following docs task and identify what changed, what is stale, and what needs to be updated.",
    "",
    "Docs task:",
    prompt,
    "",
    "Documentation context:",
    context,
  }, "\n")

  return {
    context = context,
    review_prompt = review_prompt,
    draft_prompt_prefix = table.concat({
      "Draft an operator-facing documentation update summary for the following docs task.",
      "Use the review findings as grounding.",
      "",
      "Docs task:",
      prompt,
    }, "\n"),
  }
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local built = build_context(prompt)
  local reviewer = runtime.agent("docs_reviewer")
  local drafter = runtime.agent("draft_writer")

  local review = reviewer:ask(built.review_prompt)
  if review == nil or review == "" then
    error("docs reviewer returned empty output")
  end

  local draft = drafter:ask(table.concat({
    built.draft_prompt_prefix,
    "",
    "Review findings:",
    review,
  }, "\n"))
  if draft == nil or draft == "" then
    error("draft writer returned empty output")
  end

  local brief = table.concat({
    "# Docs Team Assistant Brief",
    "",
    "Docs task: " .. prompt,
    "Generated at: " .. time.now_utc(),
    "",
    "## Review Findings",
    review,
    "",
    "## Draft Update",
    draft,
  }, "\n")

  session.incr("docs_team_assistant.run_count")
  session.set("docs_team_assistant.last_prompt", prompt)
  session.set("docs_team_assistant.review_bytes", tostring(#review))
  session.set("docs_team_assistant.draft_bytes", tostring(#draft))

  runtime.db.with("state", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS docs_team_runs (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        review TEXT NOT NULL,
        draft TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])
    db:exec(
      "INSERT INTO docs_team_runs(prompt, review, draft, created_at) VALUES (?, ?, ?, ?)",
      { prompt, review, draft, time.now_utc() }
    )
  end)

  write_required(".turin/runtime/docs-team/context.md", built.context)
  write_required(".turin/runtime/docs-team/review.md", review)
  write_required(".turin/runtime/docs-team/draft.md", draft)
  write_required(".turin/runtime/docs-team/brief.md", brief)
  write_required(".turin/runtime/docs-team/last-prompt.txt", prompt)

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Documentation context:",
    built.context,
    "",
    "Review findings:",
    review,
    "",
    "Draft update:",
    draft,
  }, "\n")

  return ALLOW
end
