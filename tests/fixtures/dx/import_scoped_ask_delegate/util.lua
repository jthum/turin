return {
  run = function()
    local reviewer = runtime.agent("reviewer")
    local status = reviewer:status()
    if status == nil or status.agent_id ~= "reviewer" then
      error("reviewer status mismatch inside delegated import")
    end

    local review = reviewer:ask("Review delegated import flow")
    runtime.db.with(".turin/runtime/harness.db", function(db)
      db:exec([[
        CREATE TABLE IF NOT EXISTS delegated_ask_probe (
          id INTEGER PRIMARY KEY,
          review TEXT NOT NULL
        )
      ]])

      db:exec(
        "INSERT INTO delegated_ask_probe(review) VALUES (?)",
        { review }
      )
    end)

    local db = access.check("db.exec")
    local policy = access.check("policy.set")
    local policy_ok, policy_err = try(runtime.policy.set, "dx.import.ask", true)

    return {
      review = review,
      db = db,
      policy = policy,
      policy_ok = policy_ok,
      policy_err = policy_err,
    }
  end
}
