function on_turn_prepare(ctx)
  local query = access.check("runtime.db.query")
  if query == nil or not query.allowed then
    error("delegated reviewer should have runtime.db.query allowed")
  end

  local rows, qerr = runtime.db.query("SELECT 7 AS n")
  if rows == nil then
    error("delegated reviewer runtime.db.query failed: " .. tostring(qerr))
  end
  if #rows < 1 or rows[1].n ~= 7 then
    error("delegated reviewer runtime.db.query mismatch")
  end

  local exec = access.check("runtime.db.exec")
  if exec == nil or exec.allowed then
    error("delegated reviewer should have runtime.db.exec denied")
  end

  local changed, err = runtime.db.exec("CREATE TABLE IF NOT EXISTS peer_complete_forbidden (id INTEGER)")
  if changed ~= nil or err == nil then
    error("delegated reviewer runtime.db.exec should be denied")
  end
  if not tostring(err):find("delegated capabilities", 1, true) then
    error("delegated reviewer runtime.db.exec denial should mention delegated capabilities")
  end

  return ALLOW
end
