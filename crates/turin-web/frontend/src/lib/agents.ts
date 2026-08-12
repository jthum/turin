import type { AgentSummary, TurinStatus } from "./types";

export function effectiveAgents(status: TurinStatus): AgentSummary[] {
  const registered = new Map(
    status.snapshot.status.registry.agents
      .map(agent => [agent.id, agent]),
  );

  return status.snapshot.status.agent_runtimes.filter(runtime => {
    const metadata = registered.get(runtime.agent_id);
    return metadata?.enabled !== false;
  }).map(runtime => {
    const metadata = registered.get(runtime.agent_id);
    return {
      id: runtime.agent_id,
      enabled: true,
      provider: runtime.provider || metadata?.provider || "",
      model: runtime.model || metadata?.model || "",
      harness_ref: runtime.harness_id || metadata?.harness_ref || "default",
    };
  });
}
