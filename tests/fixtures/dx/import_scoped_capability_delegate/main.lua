function on_turn_prepare(ctx)
  local util = import_scoped("util", {
    root = "core",
    capabilities = {
      ["runtime.db.query"] = true
    }
  })

  local result = util.inspect()
  if result.query == nil or not result.query.allowed then
    error("delegated runtime.db.query should stay allowed")
  end
  if result.policy == nil or result.policy.allowed then
    error("delegated runtime.policy.set should be denied")
  end
  if result.ok then
    error("runtime.policy.set should fail inside delegated import")
  end
  if result.err == nil or not tostring(result.err):find("delegated capabilities", 1, true) then
    error("delegation denial should mention delegated capabilities")
  end

  local self_policy = access.check("runtime.policy.set")
  if self_policy == nil or not self_policy.allowed then
    error("caller capability context should be restored after import")
  end

  return ALLOW
end
