function on_turn_prepare(ctx)
  runtime.db.with("state", function(db)
    db:exec("CREATE TABLE IF NOT EXISTS dx_journal(id INTEGER PRIMARY KEY, note TEXT)")
    db:exec("DELETE FROM dx_journal")
    db:exec("INSERT INTO dx_journal(note) VALUES (?)", { "seed" })

    local row = db:one("SELECT note FROM dx_journal ORDER BY id DESC LIMIT 1")
    if row == nil or row.note ~= "seed" then
      error("db journal mismatch")
    end
  end)

  return ALLOW
end
