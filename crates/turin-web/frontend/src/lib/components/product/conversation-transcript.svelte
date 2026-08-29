<script lang="ts">
	import { Bot, ChevronDown, ChevronUp, LoaderCircle, Plus, Sparkles } from '@lucide/svelte';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as ScrollArea from '#lib/components/ui/scroll-area/index.js';
	import type { ConversationMessage, Session } from '#lib/api/contracts.js';
	import ConversationMessageView from './conversation-message.svelte';

	let {
		ref = $bindable(), session, messages, loading, loadingWindow, hasOlder, hasNewer,
		messageTotal, submitting, streamMessageId, onLoadOlder, onLoadNewer, onCreate
	}: {
		ref: HTMLElement | null;
		session: Session | null;
		messages: ConversationMessage[];
		loading: boolean;
		loadingWindow: boolean;
		hasOlder: boolean;
		hasNewer: boolean;
		messageTotal: number;
		submitting: boolean;
		streamMessageId: string | null;
		onLoadOlder: () => void;
		onLoadNewer: () => void;
		onCreate: () => void;
	} = $props();

</script>

<ScrollArea.Root bind:viewportRef={ref} class="min-h-0 h-full" scrollbarYClasses="py-2">
	{#if loading}
		<div class="flex h-full min-h-96 items-center justify-center gap-2 text-sm text-muted-foreground"><LoaderCircle class="size-4 animate-spin" />Loading conversation</div>
	{:else if !session}
		<div class="flex h-full min-h-96 flex-col items-center justify-center px-6 text-center">
			<div class="mb-5 flex size-12 items-center justify-center rounded-2xl border bg-card text-primary shadow-sm"><Sparkles class="size-5" /></div>
			<h1 class="text-2xl font-semibold tracking-tight">What should we work on?</h1>
			<p class="mb-6 mt-2 max-w-sm text-sm text-muted-foreground">Start a durable conversation with one of your configured agents.</p>
			<Button onclick={onCreate}><Plus class="size-4" />New conversation</Button>
		</div>
	{:else}
		<div class="mx-auto w-full max-w-3xl px-4 pb-12 pt-7 sm:px-6">
			<div class="mb-6 flex justify-center">
				{#if hasOlder}
					<Button variant="outline" size="sm" onclick={onLoadOlder} disabled={loadingWindow} class="rounded-full text-muted-foreground">
						{#if loadingWindow}<LoaderCircle class="size-3.5 animate-spin" />{:else}<ChevronUp class="size-3.5" />{/if}Load earlier messages
					</Button>
				{:else if messageTotal > 0}<span class="text-xs text-muted-foreground">Beginning of conversation</span>{/if}
			</div>

			{#if messages.length === 0}
				<div class="flex min-h-96 flex-col items-center justify-center text-center">
					<div class="mb-4 flex size-11 items-center justify-center rounded-2xl border bg-card shadow-sm"><Bot class="size-5" /></div>
					<h1 class="text-xl font-semibold tracking-tight">Start with a clear outcome</h1>
					<p class="mt-2 max-w-sm text-sm text-muted-foreground">This conversation is durable, branchable, and owned by {session.agent_id}.</p>
				</div>
			{/if}

			<div class="space-y-1">
				{#each messages as message (message.id)}
					<ConversationMessageView {message} agentId={session.agent_id} streaming={streamMessageId === message.id} />
				{/each}
			</div>

			{#if submitting && !streamMessageId}
				<div class="flex items-center gap-3 py-4 pl-9 text-sm text-muted-foreground sm:pl-10">
					<span class="flex gap-1"><i class="size-1.5 animate-bounce rounded-full bg-current [animation-delay:-.3s]"></i><i class="size-1.5 animate-bounce rounded-full bg-current [animation-delay:-.15s]"></i><i class="size-1.5 animate-bounce rounded-full bg-current"></i></span>
					{session.agent_id} is thinking
				</div>
			{/if}

			{#if hasNewer}
				<div class="mt-7 flex justify-center">
					<Button variant="outline" size="sm" onclick={onLoadNewer} disabled={loadingWindow} class="rounded-full text-muted-foreground">
						{#if loadingWindow}<LoaderCircle class="size-3.5 animate-spin" />{:else}<ChevronDown class="size-3.5" />{/if}Load newer messages
					</Button>
				</div>
			{/if}
		</div>
	{/if}
</ScrollArea.Root>
