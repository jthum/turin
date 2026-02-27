return {
  try_nested_widen = function()
    local ok, err = pcall(function()
      local child = import_scoped("child", {
        root = "core",
        capabilities = {
          ["harness.import.scoped"] = true,
          ["runtime.policy.set"] = true
        }
      })
      return child.ping()
    end)
    return ok, err
  end
}
