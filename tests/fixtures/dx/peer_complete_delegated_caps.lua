function on_turn_prepare(ctx)
  local reviewer = runtime.agent("reviewer")
  local status = reviewer:status()
  if status == nil or status.agent_id ~= "reviewer" then
    error("reviewer status mismatch")
  end

  local review = reviewer:ask("Review delegated caps", {
    timeout_ms = 5000,
    capabilities = {
      ["db.query"] = true
    }
  })

  if review ~= "REVIEW_QUERY_OK" then
    error("unexpected reviewer output: " .. tostring(review))
  end

  return ALLOW
end
