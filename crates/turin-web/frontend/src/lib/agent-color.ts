export function agentHue(name: string) {
	return Array.from(name).reduce(
		(hash, character) => ((hash * 31) + character.charCodeAt(0)) % 360,
		0
	);
}
