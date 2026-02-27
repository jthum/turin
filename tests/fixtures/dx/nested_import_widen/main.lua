function on_turn_prepare(ctx)
  local util = import_scoped("util", {
    root = "core",
    capabilities = {
      ["harness.import.scoped"] = true,
      ["runtime.db.query"] = true
    }
  })

  local ok, err = util.try_nested_widen()
  if ok then
    error("nested import should not widen delegated capabilities")
  end
  if err == nil or not tostring(err):find("cannot grant 'runtime.policy.set' beyond importer delegation", 1, true) then
    error("nested import denial should mention delegation widening")
  end

  return ALLOW
end
