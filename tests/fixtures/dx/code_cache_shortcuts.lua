function on_turn_prepare(ctx)
  remember("Compiler errors should stay concise")

  local hits = recall("compiler")
  if #hits < 1 then
    error("expected top-level recall hits")
  end

  local file = fs.read("notes.txt")
  if file ~= "cached text" then
    error("fs.read content mismatch: " .. tostring(file))
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
