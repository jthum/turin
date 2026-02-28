function on_turn_prepare(ctx)
  local reviewer = runtime.agent("reviewer")
  local prompt = tostring(ctx.prompt or "")

  local review = runtime.governance.grant({
    ttl_ms = 10000,
    capabilities = {
      ["runtime.agent.submit"] = true,
      ["runtime.agent.await"] = true,
      ["runtime.agent.status"] = true,
    }
  }, function()
    return reviewer:complete("Review this request for risk and missing checks:\n\n" .. prompt)
  end)

  if review == nil or review == "" then
    error("peer review returned empty output")
  end

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Peer review preflight:",
    review,
  }, "\n")
  return ALLOW
end
