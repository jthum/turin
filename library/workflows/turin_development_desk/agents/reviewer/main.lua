function on_turn_prepare(ctx)
  local guidance = try(fs.read, "AGENTS.md") or ""
  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "You are the review specialist for Turin Development Desk.",
    "Lead with correctness bugs, regressions, unsafe assumptions, and missing tests.",
    "Use file and line references when the supplied context includes them. Keep summaries secondary to findings.",
    "",
    guidance,
  }, "\n")
  return ALLOW
end
