function on_turn_prepare(ctx)
  session.remember("Compiler errors should be concise")
  session.set("started_at", tostring(tonumber(time.now_utc()) - 1))
  session.incr("turn_prepare_calls")

  local hits = session.recall("compiler", { limit = 3 })
  if hits == nil or #hits < 1 then
    error("expected session memory hits")
  end

  if not time.after(session.get("started_at"), 0.5) then
    error("expected elapsed session time")
  end

  return ALLOW
end
