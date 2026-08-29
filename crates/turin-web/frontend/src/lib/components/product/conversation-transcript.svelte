<script lang="ts">
	import { Bot, ChevronUp, LoaderCircle, Plus, Sparkles } from '@lucide/svelte';
	import { Button } from '#lib/components/ui/button/index.js';
	import type { ConversationMessage, Session } from '#lib/api/contracts.js';

	let {
		ref = $bindable(), session, messages, loading, loadingWindow, hasOlder, hasNewer,
		messageTotal, submitting, streamMessageId, onLoadOlder, onLoadNewer, onCreate
	}: {
		ref: HTMLDivElement | undefined;
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

	function formatTime(value: string) {
		return new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(new Date(value));
	}
</script>

<div class="transcript" bind:this={ref}>
	{#if loading}
		<div class="conversation-state"><LoaderCircle class="spin" /><span>Loading conversation</span></div>
	{:else if !session}
		<div class="conversation-state empty-state"><div class="empty-orbit"><Sparkles /></div><h1>What should we work on?</h1><p>Start a conversation with one of your configured agents.</p><Button onclick={onCreate}><Plus /> New conversation</Button></div>
	{:else}
		<div class="message-column">
			{#if hasOlder}<button class="load-window" onclick={onLoadOlder} disabled={loadingWindow}>{#if loadingWindow}<LoaderCircle class="spin" />{:else}<ChevronUp />{/if}Load earlier messages</button>{:else if messageTotal > 0}<span class="history-start">Beginning of conversation</span>{/if}
			{#if messages.length === 0}<div class="conversation-state empty-state compact"><div class="empty-orbit"><Bot /></div><h1>Start with a clear outcome</h1><p>This conversation is durable, branchable, and owned by {session.agent_id}.</p></div>{/if}
			{#each messages as message (message.id)}
				<article class="message" class:user={message.role === 'user'}>
					<div class="message-avatar">{message.role === 'user' ? 'Y' : 'T'}</div>
					<div class="message-body"><div class="message-meta"><strong>{message.role === 'user' ? 'You' : session.agent_id}</strong><time>{formatTime(message.created_at)}</time></div><div class="message-content">{message.content}</div>{#if streamMessageId === message.id}<span class="stream-caret"></span>{/if}</div>
				</article>
			{/each}
			{#if submitting && !streamMessageId}<div class="thinking"><span></span><span></span><span></span><em>{session.agent_id} is thinking</em></div>{/if}
			{#if hasNewer}<button class="load-window load-newer" onclick={onLoadNewer} disabled={loadingWindow}>{#if loadingWindow}<LoaderCircle class="spin" />{:else}<ChevronUp />{/if}Load newer messages</button>{/if}
		</div>
	{/if}
</div>

<style>
	.transcript { min-height: 0; overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable; }
	.message-column { width: min(760px, calc(100% - 48px)); margin: 0 auto; padding: 28px 0 42px; }
	.message { display: grid; grid-template-columns: 28px minmax(0, 1fr); gap: 12px; padding: 16px 0; }
	.message-avatar { display: grid; width: 27px; height: 27px; place-items: center; border: 1px solid #deded9; border-radius: 8px; background: white; color: #252523; font-size: 10px; font-weight: 750; box-shadow: 0 1px 2px #0000000a; }
	.message.user .message-avatar { border-color: #242422; background: #242422; color: white; }
	.message-body { min-width: 0; }
	.message-meta { display: flex; align-items: baseline; gap: 8px; margin: 3px 0 7px; }
	.message-meta strong { font-size: 12px; font-weight: 650; text-transform: capitalize; }
	.message-meta time { color: #a0a09a; font-size: 10px; }
	.message-content { color: #333330; font-size: 14px; line-height: 1.72; white-space: pre-wrap; overflow-wrap: anywhere; }
	.message.user .message-content { display: inline-block; border: 1px solid #e5e5e0; border-radius: 4px 14px 14px 14px; background: #f3f3f0; padding: 10px 13px; line-height: 1.55; }
	.stream-caret { display: inline-block; width: 5px; height: 15px; margin-left: 2px; vertical-align: text-bottom; background: #272725; animation: blink .8s infinite; }
	.load-window { display: flex; align-items: center; gap: 6px; margin: 0 auto 18px; border: 1px solid #e1e1dc; border-radius: 999px; background: white; padding: 7px 11px; color: #6d6d67; font-size: 11px; cursor: pointer; box-shadow: 0 1px 1px #00000008; }
	.load-window :global(svg) { width: 13px; }
	.load-newer { margin-top: 22px; margin-bottom: 0; }
	.load-newer :global(svg) { transform: rotate(180deg); }
	.history-start { display: block; padding-bottom: 12px; color: #aaa9a3; font-size: 10px; text-align: center; }
	.thinking { display: flex; align-items: center; gap: 4px; padding: 16px 40px; color: #8e8e88; }
	.thinking span { width: 4px; height: 4px; border-radius: 50%; background: #8e8e88; animation: pulse 1s infinite; }
	.thinking span:nth-child(2) { animation-delay: .15s; }.thinking span:nth-child(3) { animation-delay: .3s; }
	.thinking em { margin-left: 5px; font-size: 11px; font-style: normal; }
	.conversation-state { display: flex; height: 100%; align-items: center; justify-content: center; gap: 9px; color: #85857f; font-size: 12px; }
	.empty-state { flex-direction: column; text-align: center; }
	.empty-state.compact { min-height: 420px; }
	.empty-orbit { display: grid; width: 42px; height: 42px; place-items: center; margin-bottom: 6px; border: 1px solid #dfdfda; border-radius: 13px; background: white; box-shadow: 0 8px 30px #0000000a; }
	.empty-orbit :global(svg) { width: 18px; }
	.empty-state h1 { margin: 0; color: #222220; font-size: 21px; font-weight: 620; letter-spacing: -.025em; }
	.empty-state p { max-width: 380px; margin: 6px 0 16px; color: #85857f; font-size: 12px; }
	:global(.spin) { animation: spin 1s linear infinite; }
	@keyframes spin { to { transform: rotate(360deg); } }
	@keyframes blink { 50% { opacity: 0; } }
	@keyframes pulse { 50% { opacity: .25; transform: translateY(-2px); } }
	@media (max-width: 760px) {
		.message-column { width: calc(100% - 30px); padding-top: 18px; }
		.message { grid-template-columns: 25px minmax(0, 1fr); gap: 9px; }
		.message-avatar { width: 25px; height: 25px; }
		.message-content { font-size: 13px; }
	}
</style>
