function on_turn_prepare(ctx)
  local query = access.check("db.query")
  if query == nil or not query.allowed then
    return REJECT, "delegated runtime.db.query should be allowed"
  end

  local rows, qerr = runtime.db.query("SELECT 7 AS n")
  if rows == nil then
    return REJECT, "delegated runtime.db.query failed: " .. tostring(qerr)
  end
  if rows[1] == nil or rows[1].n ~= 7 then
    return REJECT, "delegated runtime.db.query mismatch"
  end

  local exec = access.check("db.exec")
  if exec == nil or exec.allowed then
    return REJECT, "delegated runtime.db.exec should be denied"
  end

  local changed, err = runtime.db.exec("CREATE TABLE IF NOT EXISTS delegated_peer_forbidden (id INTEGER)")
  if changed ~= nil or err == nil then
    return REJECT, "delegated runtime.db.exec should be denied"
  end
  if not tostring(err):find("delegated capabilities", 1, true) then
    return REJECT, "delegated runtime.db.exec denial should mention delegated capabilities"
  end

  return ALLOW
end
