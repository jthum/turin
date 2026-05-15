action.define_on("project", "review", function(this, project, params)
  local reviews = worklist("reviews", { scope = project })
  local branch = tostring(params.branch or "main")

  return reviews:add({
    title = "Review " .. branch,
    action = "review.branch",
    params = {
      project = ref(project),
      branch = branch,
      requested_by = params.requested_by,
    },
    metadata = {
      queue = "code-review",
    },
  })
end)

action.define_on(target.workitem("projects"), "label", function(this, item, params)
  local metadata = item.metadata or {}
  metadata.label = params.label
  return item:update({ metadata = metadata })
end)

action.define("projects.snapshot", function(this, params)
  local item = params.item
  return {
    id = item.id,
    title = item.title,
    label = item.metadata and item.metadata.label or nil,
    status = item.status,
  }
end)

runtime.on("project.ready", function(data, meta)
  local project = data.project
  local item = data.item

  if type(project.review) ~= "function" then
    error("expected project:review(...) method from action.define_on")
  end
  if type(item.label) ~= "function" then
    error("expected hydrated work item proxy with object-scoped method")
  end

  item:label({ label = "ready" })
  project:review({
    branch = data.branch,
    requested_by = meta and meta.source_agent_id or "local",
  })
end)

function on_turn_prepare(turn)
  local project = scope("project", "alpha")
  project:set("owner", "planner")

  local projects = worklist("projects", { scope = project })
  local item = projects:add({
    title = "Prepare launch checklist",
    prompt = "Prepare launch checklist",
    metadata = {
      phase = "planning",
    },
  })

  runtime.emit("project.ready", {
    project = ref(project),
    item = item,
    branch = "feature-x",
  })

  local snapshot = action.run("projects.snapshot", {
    item = ref(item),
  })

  session.set("reference_example.last_item_id", snapshot.id)
  session.set("reference_example.last_item_label", tostring(snapshot.label or ""))
  session.set("reference_example.last_project_owner", tostring(project:get("owner") or ""))

  return ALLOW
end
