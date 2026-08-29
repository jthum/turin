import type { Bootstrap } from './contracts.js';

export type { Bootstrap, RuntimeHealth } from './contracts.js';

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
