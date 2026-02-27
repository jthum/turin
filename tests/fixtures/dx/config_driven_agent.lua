function on_turn_prepare(ctx)
  local cfg = fs.read_json("config/agent.json")
  if cfg.mode ~= "draft" then
    error("config mode mismatch")
  end

  cfg.touches = (cfg.touches or 0) + 1
  cfg.last_seen = time.now_utc()

  fs.write_json("config/agent.json", cfg, { pretty = true })
  session.incr("config_touches")

  return ALLOW
end
