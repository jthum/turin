<script lang="ts">
	import { tick, untrack } from 'svelte';
	import { createVirtualizer } from '@tanstack/svelte-virtual';
	import { ArrowDown, Bot, LoaderCircle, Plus, Sparkles } from '@lucide/svelte';
	import { Button } from '#lib/components/ui/button/index.js';
	import * as ScrollArea from '#lib/components/ui/scroll-area/index.js';
	import * as Tooltip from '#lib/components/ui/tooltip/index.js';
	import type { ConversationMessage, Session } from '#lib/api/contracts.js';
	import ConversationMap from './conversation-map.svelte';
	import ConversationMessageView from './conversation-message.svelte';

	let {
		ref = $bindable(), session, agentName, messages, loading, loadingWindow, hasOlder, hasNewer,
		messageTotal, newestOffset, olderOffset, submitting, streamMessageId,
		onLoadOlder, onLoadNewer, onJumpToPosition, onJumpToEnd, onFork, onCreate
	}: {
		ref: HTMLElement | null;
		session: Session | null;
		agentName: string;
		messages: ConversationMessage[];
		loading: boolean;
		loadingWindow: boolean;
		hasOlder: boolean;
		hasNewer: boolean;
		messageTotal: number;
		newestOffset: number;
		olderOffset: number;
		submitting: boolean;
		streamMessageId: string | null;
		onLoadOlder: () => void;
		onLoadNewer: () => void;
		onJumpToPosition: (position: number) => void;
		onJumpToEnd: () => void;
		onFork: (message: ConversationMessage, activate: boolean) => void;
		onCreate: () => void;
	} = $props();
	let showJumpToEnd = $state(false);
	let viewportProgress = $state(1);

	const virtualizer = createVirtualizer<HTMLElement, HTMLElement>({
		count: 0,
		getScrollElement: () => ref,
		getItemKey: (index) => messages[index]?.id ?? index,
		estimateSize: () => 112,
		overscan: 7
	});

	$effect(() => {
		const count = messages.length;
		const scrollElement = ref;
		untrack(() => {
			$virtualizer.setOptions({
				count,
				getScrollElement: () => scrollElement,
				getItemKey: (index) => messages[index]?.id ?? index,
				estimateSize: () => 112,
				overscan: 7
			});
		});
	});

	$effect(() => {
		const viewport = ref;
		const canLoadOlder = hasOlder;
		const canLoadNewer = hasNewer;
		const busy = loading || loadingWindow;
		if (!viewport || busy) return;

		function loadAtBoundary() {
			const distanceFromBottom = viewport!.scrollHeight - viewport!.clientHeight - viewport!.scrollTop;
			const scrollableDistance = Math.max(1, viewport!.scrollHeight - viewport!.clientHeight);
			viewportProgress = Math.min(1, Math.max(0, viewport!.scrollTop / scrollableDistance));
			showJumpToEnd = canLoadNewer || distanceFromBottom > 320;
			if (canLoadOlder && viewport!.scrollTop < 480) {
				onLoadOlder();
				return;
			}
			if (canLoadNewer && distanceFromBottom < 480) onLoadNewer();
		}

		viewport.addEventListener('scroll', loadAtBoundary, { passive: true });
		const frame = requestAnimationFrame(loadAtBoundary);
		return () => {
			cancelAnimationFrame(frame);
			viewport.removeEventListener('scroll', loadAtBoundary);
		};
	});

	function measureMessage(node: HTMLElement) {
		$virtualizer.measureElement(node);
	}

	export async function restoreMessageAnchor(id: string, viewportOffset: number) {
		await tick();
		const index = messages.findIndex((message) => message.id === id);
		if (index < 0 || !ref) return;
		$virtualizer.scrollToIndex(index, { align: 'start' });
		await tick();
		await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
		const element = Array.from(ref.querySelectorAll<HTMLElement>('[data-message-id]'))
			.find((candidate) => candidate.dataset.messageId === id);
		if (!element) return;
		const currentOffset = element.getBoundingClientRect().top - ref.getBoundingClientRect().top;
		ref.scrollTop += currentOffset - viewportOffset;
	}

	export async function scrollToEnd(behavior: ScrollBehavior = 'instant') {
		await tick();
		if (messages.length > 0) $virtualizer.scrollToIndex(messages.length - 1, { align: 'end' });
		await tick();
		await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
		ref?.scrollTo({ top: ref.scrollHeight, behavior });
	}
</script>

<div class="relative h-full min-h-0">
<ScrollArea.Root bind:viewportRef={ref} class="min-h-0 h-full bg-background" scrollbarYClasses="py-2">
	{#if loading}
		<div class="flex h-full min-h-96 items-center justify-center gap-2 text-sm text-muted-foreground"><LoaderCircle class="size-4 animate-spin" />Loading conversation</div>
	{:else if !session}
		<div class="flex h-full min-h-96 flex-col items-center justify-center px-6 text-center">
			<div class="mb-5 flex size-12 items-center justify-center rounded-2xl border bg-background text-primary"><Sparkles class="size-5" /></div>
			<h1 class="text-2xl font-semibold tracking-tight">What should we work on?</h1>
			<p class="mb-6 mt-2 max-w-sm text-sm text-muted-foreground">Start a durable conversation with one of your configured agents.</p>
			<Button onclick={onCreate}><Plus class="size-4" />New conversation</Button>
		</div>
	{:else}
		<div class="mx-auto w-full max-w-3xl px-4 pb-12 pt-7 sm:px-6">
			{#if messages.length === 0}
				<div class="flex min-h-96 flex-col items-center justify-center text-center">
					<div class="mb-4 flex size-11 items-center justify-center rounded-2xl border bg-background"><Bot class="size-5" /></div>
					<h1 class="text-xl font-semibold tracking-tight">Start with a clear outcome</h1>
					<p class="mt-2 max-w-sm text-sm text-muted-foreground">This conversation is durable and branchable. {agentName} is ready when you are.</p>
				</div>
			{/if}

			{#if loadingWindow}
				<div class="pointer-events-none sticky top-3 z-10 flex justify-center" aria-live="polite">
					<span class="flex items-center gap-2 rounded-full border bg-background px-3 py-1.5 text-xs text-muted-foreground"><LoaderCircle class="size-3 animate-spin" />Loading history</span>
				</div>
			{/if}

			<div class="relative w-full" style:height={`${$virtualizer.getTotalSize()}px`}>
			{#each $virtualizer.getVirtualItems() as row (row.key)}
				{@const message = messages[row.index]}
				{#if message}
					<div
						use:measureMessage
						data-index={row.index}
						data-message-id={message.id}
						class="absolute left-0 top-0 w-full"
						style:transform={`translateY(${row.start}px)`}
					>
						<ConversationMessageView {message} {agentName} streaming={streamMessageId === message.id} {onFork} />
					</div>
				{/if}
			{/each}
			</div>

			{#if submitting && !streamMessageId}
				<div class="flex items-center gap-3 py-4 pl-9 text-sm text-muted-foreground sm:pl-10">
					<span class="flex gap-1"><i class="size-1.5 animate-bounce rounded-full bg-current [animation-delay:-.3s]"></i><i class="size-1.5 animate-bounce rounded-full bg-current [animation-delay:-.15s]"></i><i class="size-1.5 animate-bounce rounded-full bg-current"></i></span>
					{agentName} is thinking
				</div>
			{/if}

		</div>
	{/if}
</ScrollArea.Root>
{#if session}
	<ConversationMap {messages} total={messageTotal} {newestOffset} {olderOffset} {viewportProgress} loading={loadingWindow} onJump={onJumpToPosition} />
{/if}
{#if session && showJumpToEnd}
	<div class="pointer-events-none absolute bottom-4 left-1/2 z-20 -translate-x-1/2">
		<Tooltip.Root>
			<Tooltip.Trigger>
				{#snippet child({ props })}
					<Button {...props} variant="outline" size="icon" class="pointer-events-auto rounded-full bg-background" onclick={onJumpToEnd} aria-label="Jump to latest message"><ArrowDown class="size-4" /></Button>
				{/snippet}
			</Tooltip.Trigger>
			<Tooltip.Content>Jump to latest message</Tooltip.Content>
		</Tooltip.Root>
	</div>
{/if}
</div>
