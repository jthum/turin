local function read_required(path)
  local text, err = fs.read(path)
  if not text then
    error("required bug triage file missing: " .. path .. ": " .. tostring(err))
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
    error("failed to write bug triage artifact " .. path .. ": " .. tostring(err))
  end
end

local function build_context(prompt)
  local severity_policy = read_required("SEVERITY_POLICY.md")
  local ownership = read_required("OWNERSHIP.md")
  local known_issues = read_required("KNOWN_ISSUES.md")
  local runbook = read_optional("RUNBOOK.md")

  local sections = {
    "# SEVERITY_POLICY.md",
    severity_policy,
    "",
    "# OWNERSHIP.md",
    ownership,
    "",
    "# KNOWN_ISSUES.md",
    known_issues,
  }

  if runbook and runbook ~= "" then
    table.insert(sections, "")
    table.insert(sections, "# RUNBOOK.md")
    table.insert(sections, runbook)
  end

  local context = table.concat(sections, "\n")
  local triage_prompt = table.concat({
    "Classify the following bug report.",
    "Return a practical triage summary covering severity, likely owner, subsystem, reproduction clarity, and next checks.",
    "",
    "Bug report:",
    prompt,
    "",
    "Workspace triage context:",
    context,
  }, "\n")

  return {
    context = context,
    triage_prompt = triage_prompt,
    response_prompt_prefix = table.concat({
      "Draft an operator-facing response and next-action checklist for the following bug report.",
      "Keep it practical and aligned with the triage outcome.",
      "",
      "Bug report:",
      prompt,
    }, "\n"),
  }
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local built = build_context(prompt)
  local triager = runtime.agent("triager")
  local responder = runtime.agent("responder")

  local triage = triager:ask(built.triage_prompt)
  if triage == nil or triage == "" then
    error("triager returned empty output")
  end

  local response = responder:ask(table.concat({
    built.response_prompt_prefix,
    "",
    "Triage summary:",
    triage,
  }, "\n"))
  if response == nil or response == "" then
    error("responder returned empty output")
  end

  local brief = table.concat({
    "# Bug Triage Desk Brief",
    "",
    "Bug report: " .. prompt,
    "Generated at: " .. time.now_utc(),
    "",
    "## Triage",
    triage,
    "",
    "## Response",
    response,
  }, "\n")

  session.incr("bug_triage.run_count")
  session.set("bug_triage.last_prompt", prompt)
  session.set("bug_triage.triage_bytes", tostring(#triage))
  session.set("bug_triage.response_bytes", tostring(#response))

  runtime.db.with("state", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS bug_triage_runs (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        triage TEXT NOT NULL,
        response TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])

    db:exec(
      "INSERT INTO bug_triage_runs(prompt, triage, response, created_at) VALUES (?, ?, ?, ?)",
      { prompt, triage, response, time.now_utc() }
    )
  end)

  write_required(".turin/runtime/bug-triage/context.md", built.context)
  write_required(".turin/runtime/bug-triage/triage.md", triage)
  write_required(".turin/runtime/bug-triage/response.md", response)
  write_required(".turin/runtime/bug-triage/brief.md", brief)
  write_required(".turin/runtime/bug-triage/last-prompt.txt", prompt)

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Bug triage workspace context:",
    built.context,
    "",
    "Triage summary:",
    triage,
    "",
    "Operator response draft:",
    response,
  }, "\n")

  return ALLOW
end
