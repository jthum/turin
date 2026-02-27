function on_turn_prepare(ctx)
  if not allowed("runtime.db.query") then
    error("runtime.db.query should be allowed")
  end

  local dec = access.check("runtime.db.exec")
  if dec == nil then
    error("missing runtime.db.exec decision")
  end
  if dec.allowed ~= false then
    error("runtime.db.exec should be denied in governed mode")
  end

  local ok, err = pcall(function()
    needs("runtime.db.exec")
  end)
  if ok then
    error("needs should raise on denied capability")
  end
  if not tostring(err):find("Governance denial", 1, true) then
    error("needs denial should surface governance reason")
  end

  return ALLOW
end
