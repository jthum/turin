return {
  inspect = function()
    local query = access.check("runtime.db.query")
    local policy = access.check("runtime.policy.set")
    local ok, err = runtime.policy.set("dx.import.flag", true)
    return {
      query = query,
      policy = policy,
      ok = ok,
      err = err,
    }
  end
}
