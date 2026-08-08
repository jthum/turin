function on_turn_prepare(ctx)
  local guidance = try(fs.read, "AGENTS.md") or ""
  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "You are the planning specialist for Turin Development Desk.",
    "Produce a concrete, repository-aware plan with likely files, sequencing, risks, and focused validation.",
    "Prefer the smallest coherent vertical slice. Do not invent abstractions without evidence from the codebase.",
    "",
    guidance,
  }, "\n")
  return ALLOW
end
