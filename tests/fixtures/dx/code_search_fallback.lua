function on_turn_prepare(ctx)
  local status = runtime.code.search.status(".")
  if status.capabilities == nil or status.capabilities.lexical ~= true then
    error("expected lexical capability")
  end
  if status.capabilities.semantic ~= false then
    error("expected lexical-only index for fallback fixture")
  end

  local rows = code.find("capability decision", {
    trace = true,
    strict = false,
    limit = 3,
  })
  if rows == nil or #rows < 1 then
    error("expected fallback rows")
  end
  if rows[1].name ~= "capability_decision" then
    error("unexpected fallback row: " .. tostring(rows[1].name))
  end

  local trace = rows[1].trace
  if trace == nil then
    error("expected fallback trace metadata")
  end
  if trace.requested_mode ~= "hybrid" then
    error("expected requested hybrid mode, got " .. tostring(trace.requested_mode))
  end
  if trace.effective_mode ~= "lexical" then
    error("expected lexical fallback, got " .. tostring(trace.effective_mode))
  end
  if trace.fallback_reason ~= "capability_fallback" then
    error("unexpected fallback reason: " .. tostring(trace.fallback_reason))
  end

  return ALLOW
end
