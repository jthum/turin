return {
  run = function()
    local reviewer = runtime.agent("reviewer")
    local status = reviewer:status()
    if status == nil or status.agent_id ~= "reviewer" then
      error("reviewer status mismatch inside delegated import")
    end

    local review = reviewer:ask("Review delegated import flow")
    local changed, derr = runtime.db.exec([[
      CREATE TABLE IF NOT EXISTS delegated_complete_probe (
        id INTEGER PRIMARY KEY,
        review TEXT NOT NULL
      )
    ]])
    if changed == nil then
      error("runtime.db.exec create after delegated runtime.agent.ask failed: " .. tostring(derr))
    end

    changed, derr = runtime.db.exec(
      "INSERT INTO delegated_complete_probe(review) VALUES (?)",
      { review }
    )
    if changed == nil then
      error("runtime.db.exec insert after delegated runtime.agent.ask failed: " .. tostring(derr))
    end

    local db = access.check("db.exec")
    local policy = access.check("policy.set")
    local policy_ok, policy_err = runtime.policy.set("dx.import.complete", true)

    return {
      review = review,
      db = db,
      policy = policy,
      policy_ok = policy_ok,
      policy_err = policy_err,
    }
  end
}
