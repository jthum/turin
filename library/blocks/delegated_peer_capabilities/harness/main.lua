function on_turn_prepare(ctx)
  local reviewer = runtime.agent("reviewer")
  local prompt = tostring(ctx.prompt or "")

  local review = reviewer:ask("Review this request with read-only DB access:\n\n" .. prompt, {
    timeout_ms = 10000,
    capabilities = {
      ["db.query"] = true
    }
  })

  if review == nil or review == "" then
    error("delegated peer review returned empty output")
  end

  local ok, err = try(fs.write, ".turin/runtime/delegated-review.txt", review)
  if not ok then
    error("failed to write delegated review artifact: " .. tostring(err))
  end

  ok, err = try(fs.write, ".turin/runtime/delegated-review-input.txt", prompt)
  if not ok then
    error("failed to write delegated review input artifact: " .. tostring(err))
  end

  ctx.system_prompt = table.concat({
    ctx.system_prompt or "",
    "",
    "Delegated reviewer preflight:",
    review,
  }, "\n")
  return ALLOW
end
