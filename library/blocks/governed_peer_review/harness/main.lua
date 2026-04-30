function on_turn_prepare(ctx)
  local reviewer = runtime.agent("reviewer")
  local prompt = tostring(ctx.prompt or "")

  local review = runtime.governance.grant({
    ttl_ms = 10000,
    capabilities = {
      ["agent.submit"] = true,
      ["agent.await"] = true,
      ["agent.status"] = true,
    }
  }, function()
    return reviewer:ask("Review this request for risk and missing checks:\n\n" .. prompt)
  end)

  if review == nil or review == "" then
    error("peer review returned empty output")
  end

  local ok, err = try(fs.write, ".turin/runtime/peer-review.txt", review)
  if not ok then
    error("failed to write peer review artifact: " .. tostring(err))
  end

  ok, err = try(fs.write, ".turin/runtime/peer-review-input.txt", prompt)
  if not ok then
    error("failed to write peer review input artifact: " .. tostring(err))
  end

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Peer review preflight:",
    review,
  }, "\n")
  return ALLOW
end
