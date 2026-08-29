export type RuntimeHealth = {
	connection_kind: 'local' | 'remote';
	ready: boolean;
	version: string;
	protocol_version: number;
	issue_count: number;
	agent_count: number;
	harness_count: number;
	running_agent_count: number;
	active_task_count: number;
};

export type Bootstrap = {
	web_version: string;
	runtime: RuntimeHealth;
};

export async function loadBootstrap(signal?: AbortSignal): Promise<Bootstrap> {
	const response = await fetch('/api/bootstrap', {
		headers: { Accept: 'application/json' },
		signal
	});
	if (!response.ok) {
		throw new Error(`Turin returned ${response.status}`);
	}
	return response.json() as Promise<Bootstrap>;
}
