function on_turn_prepare(ctx)
  local blocked = runtime.agent("blocked")
  local ok, err = pcall(function()
    return blocked:complete("This should be denied")
  end)

  if ok then
    error("blocked child agent should not be allowed")
  end
  if err == nil or not tostring(err):find("allowed_child_agents", 1, true) then
    error("child agent denial should mention allowed_child_agents")
  end

  return ALLOW
end
