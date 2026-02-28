function on_turn_prepare(ctx)
  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Return a concise review with risks, missing tests, and next steps.",
  }, "\n")
  return ALLOW
end
