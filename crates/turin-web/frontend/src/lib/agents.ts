import type { AgentSummary, TurinStatus } from "./types";

export function effectiveAgents(status: TurinStatus): AgentSummary[] {
  const registered = new Map(
    status.snapshot.status.registry.agents
      .filter(agent => agent.enabled)
      .map(agent => [agent.id, agent]),
  );

  return status.snapshot.status.agent_runtimes.map(runtime => registered.get(runtime.agent_id) ?? {
    id: runtime.agent_id,
    enabled: true,
    provider: "",
    model: "",
    harness_ref: runtime.agent_id === "default" ? "default" : "",
  });
}
