function on_turn_prepare(ctx)
  remember("Workspace reviews should mention safety and DX")

  local readme = fs.read("README.md")
  if not string.find(readme, "DX review fixture", 1, true) then
    error("fs.read missing README content")
  end

  local status = runtime.code.search.status(".")
  if status.capabilities == nil or status.capabilities.lexical ~= true then
    error("expected lexical capability")
  end

  local rows = code.find("capability decision", { limit = 3 })
  if rows == nil or #rows < 1 then
    error("expected code.find hits")
  end
  if rows[1].name ~= "capability_decision" then
    error("unexpected code.find top hit: " .. tostring(rows[1].name))
  end

  session.set("top_symbol", rows[1].name)
  local recalled = recall("safety", { limit = 2 })
  if recalled == nil or #recalled < 1 then
    error("expected recall hits")
  end
  if session.get("top_symbol") ~= "capability_decision" then
    error("session state mismatch")
  end

  return ALLOW
end
