function on_turn_prepare(ctx)
  local prompt = tostring(ctx.prompt or "")

  session.remember(prompt, { kind = "journal_prompt" })

  runtime.db.with(".turin/runtime/harness.db", function(db)
    db:exec([[
      CREATE TABLE IF NOT EXISTS example_journal (
        id INTEGER PRIMARY KEY,
        prompt TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
    ]])
    db:exec(
      "INSERT INTO example_journal(prompt, created_at) VALUES (?, ?)",
      { prompt, time.now_utc() }
    )

    local row = db:one("SELECT prompt FROM example_journal ORDER BY id DESC LIMIT 1")
    if row == nil or row.prompt ~= prompt then
      error("journal row mismatch")
    end

    fs.write(".turin/runtime/journal-last.txt", row.prompt)
  end)

  return ALLOW
end
