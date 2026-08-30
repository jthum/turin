export type WorkspaceSection = 'overview' | 'conversations' | 'work' | 'memory' | 'agents' | 'settings';

export const workspaceLabels: Record<WorkspaceSection, string> = {
	overview: 'Overview',
	conversations: 'Conversations',
	work: 'Work',
	memory: 'Memory',
	agents: 'Agents',
	settings: 'Settings'
};
