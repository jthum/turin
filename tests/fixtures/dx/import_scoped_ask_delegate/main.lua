function on_turn_prepare(ctx)
  local util = import_scoped("util", {
    root = "core",
    capabilities = {
      ["runtime.agent.submit"] = true,
      ["runtime.agent.await"] = true,
      ["runtime.agent.status"] = true,
      ["runtime.db.exec"] = true
    }
  })

  local result = util.run()
  if result.review ~= "REVIEW_OK" then
    error("delegated runtime.agent.ask output mismatch: " .. tostring(result.review))
  end
  if result.db == nil or not result.db.allowed then
    error("delegated runtime.db.exec should stay allowed")
  end
  if result.policy == nil or result.policy.allowed then
    error("delegated runtime.policy.set should be denied")
  end
  if result.policy_ok then
    error("runtime.policy.set should fail inside delegated import after runtime.agent.ask")
  end
  if result.policy_err == nil or not tostring(result.policy_err):find("delegated capabilities", 1, true) then
    error("delegation denial should mention delegated capabilities")
  end

  local self_policy = access.check("policy.set")
  if self_policy == nil or not self_policy.allowed then
    error("caller capability context should be restored after imported runtime.agent.ask")
  end

  return ALLOW
end
