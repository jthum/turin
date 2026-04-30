function on_turn_prepare(ctx)
  local reviewer = runtime.agent("reviewer")
  local status = reviewer:status()
  if status == nil or status.agent_id ~= "reviewer" then
    error("reviewer status mismatch")
  end

  local review = runtime.governance.grant({
    ttl_ms = 5000,
    capabilities = {
      ["agent.submit"] = true,
      ["agent.await"] = true,
      ["agent.status"] = true,
    }
  }, function()
    return reviewer:ask("Review this patch")
  end)

  if review ~= "REVIEW_OK" then
    error("unexpected reviewer output: " .. tostring(review))
  end

  return ALLOW
end
