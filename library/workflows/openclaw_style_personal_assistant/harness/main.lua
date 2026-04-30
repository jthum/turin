local function read_required(path)
  local text, err = try(fs.read, path)
  if not text then
    error("required contract file missing: " .. path .. ": " .. tostring(err))
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
    error("failed to write runtime artifact " .. path .. ": " .. tostring(err))
  end
end

local function contains_any(text, patterns)
  local haystack = string.lower(text or "")
  for _, pattern in ipairs(patterns) do
    if string.find(haystack, pattern, 1, true) then
      return true
    end
  end
  return false
end

local function choose_route(prompt)
  if contains_any(prompt, { "review", "audit", "regression", "risk", "bug" }) then
    return "reviewer"
  end

  if contains_any(prompt, { "plan", "roadmap", "task", "steps", "break down" }) then
    return "planner"
  end

  return nil
end

local function assemble_contract()
  local soul = read_required("SOUL.md")
  local profile = read_required("PROFILE.md")
  local agents = read_required("AGENTS.md")
  local inbox = read_optional("INBOX.md")

  local parts = {
    "# SOUL.md",
    soul,
    "",
    "# PROFILE.md",
    profile,
    "",
    "# AGENTS.md",
    agents,
  }

  if inbox and inbox ~= "" then
    table.insert(parts, "")
    table.insert(parts, "# INBOX.md")
    table.insert(parts, inbox)
  end

  return table.concat(parts, "\n")
end

local function build_brief(prompt, route, inbox)
  local lines = {
    "# Personal Assistant Brief",
    "",
    "Prompt: " .. prompt,
    "Route: " .. (route or "none"),
    "Generated at: " .. time.now_utc(),
    "",
    "## Inbox status",
  }

  if inbox and inbox ~= "" then
    table.insert(lines, inbox)
  else
    table.insert(lines, "No INBOX.md present.")
  end

  return table.concat(lines, "\n")
end

local function delegate(route, prompt)
  if route == "reviewer" then
    return runtime.agent("reviewer"):ask(
      "Review the following request. Focus on correctness, risk, regressions, and missing checks.\n\nUser request:\n" .. prompt
    )
  end

  if route == "planner" then
    return runtime.agent("planner"):ask(
      "Turn the following request into a concrete action plan with sequencing, dependencies, and next steps.\n\nUser request:\n" .. prompt
    )
  end

  return nil
end

function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")
  local route = choose_route(prompt)
  local contract = assemble_contract()
  local inbox = read_optional("INBOX.md")
  local brief = build_brief(prompt, route, inbox)
  local delegated_output = delegate(route, prompt)

  if route ~= nil and (delegated_output == nil or delegated_output == "") then
    error("delegated personal assistant route returned empty output")
  end

  session.incr("personal_assistant.turn_count")
  session.set("personal_assistant.route", route or "none")
  session.set("personal_assistant.last_prompt", prompt)
  session.set("personal_assistant.contract_loaded_at", time.now_utc())
  session.set("personal_assistant.contract_bytes", tostring(#contract))

  runtime.db.with("state", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS personal_assistant_activity (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        route TEXT NOT NULL,
        delegated_agent TEXT NOT NULL,
        delegated_output TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])

    db:exec(
      "INSERT INTO personal_assistant_activity(prompt, route, delegated_agent, delegated_output, created_at) VALUES (?, ?, ?, ?, ?)",
      { prompt, route or "none", route or "", delegated_output or "", time.now_utc() }
    )
  end)

  write_required(".turin/runtime/personal-assistant/contract.md", contract)
  write_required(".turin/runtime/personal-assistant/brief.md", brief)
  write_required(".turin/runtime/personal-assistant/route.txt", route or "none")
  write_required(".turin/runtime/personal-assistant/last-prompt.txt", prompt)

  if delegated_output then
    write_required(".turin/runtime/personal-assistant/delegated-output.txt", delegated_output)
  end

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Workspace contract:",
    contract,
    "",
    "Current assistant brief:",
    brief,
  }, "\n")

  if route ~= nil then
    ctx.system_prompt = table.concat({
      ctx.system_prompt,
      "",
      string.upper(route) .. " preflight:",
      delegated_output,
    }, "\n")
  end

  return ALLOW
end
