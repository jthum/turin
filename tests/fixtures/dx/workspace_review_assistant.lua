function on_turn_prepare(ctx)
  remember("Workspace reviews should mention safety and DX")

  local cached = cache.file("README.md", { include_content = true })
  if cached == nil then
    error("expected cached README")
  end
  if cached.content == nil or not string.find(cached.content, "DX review fixture", 1, true) then
    error("cache.file missing README content")
  end

  local status, serr = runtime.code.search.status(".")
  if status == nil then
    error("expected code search status: " .. tostring(serr))
  end
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
