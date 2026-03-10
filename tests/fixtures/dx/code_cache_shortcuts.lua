function on_turn_prepare(ctx)
  remember("Compiler errors should stay concise")

  local hits = recall("compiler")
  if #hits < 1 then
    error("expected top-level recall hits")
  end

  local file = cache.file("notes.txt")
  if file == nil then
    error("expected cache.file result")
  end
  if file.path ~= "notes.txt" then
    error("cache.file path mismatch: " .. tostring(file.path))
  end

  local rows = code.find("capability")
  if #rows < 1 then
    error("expected code.find rows")
  end
  if rows[1].name ~= "capability_decision" then
    error("code.find row mismatch: " .. tostring(rows[1].name))
  end

  return ALLOW
end
