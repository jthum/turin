<script lang="ts">
	import { ArrowRight, Brain, Check, ListTodo, Plus } from '@lucide/svelte';
	import type { Agent, Session } from '#lib/api/contracts.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as Card from '#lib/components/ui/card/index.js';
	import type { WorkspaceSection } from '#lib/workspace.js';
	import AgentMarker from './agent-marker.svelte';

	let {
		sessions, agents, onCreate, onSelect, onNavigate
	}: {
		sessions: Session[];
		agents: Agent[];
		onCreate: (agentId?: string) => void;
		onSelect: (session: Session) => void;
		onNavigate: (section: WorkspaceSection) => void;
	} = $props();

	let recent = $derived(sessions.slice(0, 7));
	let selectedAgentId = $state('');
	let selectedAgent = $derived(agents.find((agent) => agent.id === selectedAgentId) ?? agents[0]);

	$effect(() => {
		if (!agents.some((agent) => agent.id === selectedAgentId)) selectedAgentId = agents[0]?.id ?? '';
	});

	function relativeTime(value: string) {
		const seconds = Math.max(1, Math.floor((Date.now() - Date.parse(value)) / 1000));
		if (seconds < 3600) return `${Math.max(1, Math.floor(seconds / 60))}m ago`;
		if (seconds < 86_400) return `${Math.floor(seconds / 3600)}h ago`;
		return `${Math.floor(seconds / 86_400)}d ago`;
	}

</script>

<div class="h-full overflow-y-auto bg-muted/15">
	<div class="mx-auto flex w-full max-w-[86rem] flex-col gap-8 px-5 py-8 lg:px-8 lg:py-10">
		<header class="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
			<div>
				<h1 class="font-heading text-3xl font-semibold tracking-tight sm:text-4xl">Pick up where you left off.</h1>
				<p class="mt-2 max-w-xl text-sm leading-6 text-muted-foreground">Continue an active thread, or give an agent a new outcome to work toward.</p>
			</div>
			<Button variant="outline" onclick={() => onNavigate('conversations')}>View conversations<ArrowRight /></Button>
		</header>

		<div class="grid gap-5 xl:grid-cols-[minmax(0,1.65fr)_minmax(19rem,0.7fr)]">
			<Card.Root class="gap-0 overflow-hidden py-0 shadow-none" aria-label="Recent conversations">
				<Card.Content class="p-0">
					{#if recent.length === 0}<div class="grid min-h-72 place-items-center text-sm text-muted-foreground">Your recent work will appear here.</div>{/if}
					{#each recent as session}
						<button class="group flex w-full items-center gap-3 border-b px-5 py-4 text-left transition-colors last:border-b-0 hover:bg-muted/45" onclick={() => onSelect(session)}>
							<span class="min-w-0 flex-1">
								<span class="block truncate text-sm font-semibold text-foreground">{session.title}</span>
								{#if session.latest_message_preview}<span class="mt-1 block truncate text-sm leading-5 text-muted-foreground">{session.latest_message_preview}</span>{/if}
								<span class="mt-2 flex items-center gap-1.5 text-[11px] text-muted-foreground/75">
									<AgentMarker name={agents.find((agent) => agent.id === session.agent_id)?.name ?? session.agent_id} class="size-2" />
									<span class="font-medium text-muted-foreground">{agents.find((agent) => agent.id === session.agent_id)?.name ?? session.agent_id}</span>
									<span aria-hidden="true">·</span>
									<span>{relativeTime(session.latest_message_created_at ?? session.created_at)}</span>
								</span>
							</span>
							<ArrowRight class="size-4 text-muted-foreground/50 transition-transform group-hover:translate-x-0.5 group-hover:text-foreground" />
						</button>
					{/each}
				</Card.Content>
			</Card.Root>

			<Card.Root class="gap-0 overflow-hidden py-0 shadow-none">
				<Card.Header class="gap-1 border-b py-5"><Card.Title>Start a conversation</Card.Title><Card.Description>Choose the agent that should own the context.</Card.Description></Card.Header>
				<Card.Content class="space-y-2 p-4">
					{#each agents as agent}
						<button class={`flex w-full items-center gap-3 rounded-xl border px-3.5 py-3 text-left transition-colors hover:bg-muted/50 ${selectedAgent?.id === agent.id ? 'border-primary bg-primary/5' : ''}`} onclick={() => selectedAgentId = agent.id}>
							<AgentMarker name={agent.name} class="size-3" />
							<span class="min-w-0 flex-1"><span class="block truncate text-sm font-medium">{agent.name}</span><span class="block truncate text-xs text-muted-foreground">{agent.model}</span></span>
							{#if selectedAgent?.id === agent.id}<span class="grid size-5 place-items-center rounded-full bg-primary text-primary-foreground"><Check class="size-3" /></span>{/if}
						</button>
					{/each}
					<Button class="mt-4 w-full" size="lg" disabled={!selectedAgent} onclick={() => selectedAgent && onCreate(selectedAgent.id)}><Plus />Start with {selectedAgent?.name ?? 'agent'}</Button>
				</Card.Content>
			</Card.Root>
		</div>

		<section>
			<div class="mb-4"><h2 class="font-heading text-lg font-semibold">Beyond the conversation</h2><p class="mt-1 text-sm text-muted-foreground">Review durable work or return to knowledge Turin has retained.</p></div>
			<div class="grid gap-4 md:grid-cols-2">
				<button class="group flex items-center gap-4 rounded-2xl border bg-background p-5 text-left transition-colors hover:border-primary/25 hover:bg-primary/3" onclick={() => onNavigate('work')}>
					<span class="grid size-10 shrink-0 place-items-center rounded-xl bg-amber-500/10 text-amber-700 dark:text-amber-400"><ListTodo class="size-4" /></span>
					<span class="min-w-0 flex-1"><span class="block font-medium">Review work</span><span class="mt-1 block text-sm text-muted-foreground">See queued, active, paused, and completed items.</span></span>
					<ArrowRight class="size-4 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
				</button>
				<button class="group flex items-center gap-4 rounded-2xl border bg-background p-5 text-left transition-colors hover:border-primary/25 hover:bg-primary/3" onclick={() => onNavigate('memory')}>
					<span class="grid size-10 shrink-0 place-items-center rounded-xl bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"><Brain class="size-4" /></span>
					<span class="min-w-0 flex-1"><span class="block font-medium">Search memory</span><span class="mt-1 block text-sm text-muted-foreground">Find durable knowledge and inspect its provenance.</span></span>
					<ArrowRight class="size-4 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
				</button>
			</div>
		</section>
	</div>
</div>
