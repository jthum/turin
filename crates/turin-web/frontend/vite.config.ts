import tailwindcss from '@tailwindcss/vite';
import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import { turinMockApi } from './dev/mock-api/index.js';

const useMockApi = process.env.TURIN_WEB_MOCK === '1';

export default defineConfig({
	plugins: [
		...(useMockApi ? [turinMockApi()] : []),
		tailwindcss(),
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			adapter: adapter({ fallback: '200.html' })
		})
	],
	server: {
		proxy: useMockApi ? undefined : { '/api': 'http://127.0.0.1:9330' }
	}
});
