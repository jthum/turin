import type { Reroute } from '@sveltejs/kit/hooks';

const workspacePath = /^\/(?:conversations(?:\/[^/]+)?|work|memory|agents|settings)\/?$/;

export const reroute: Reroute = ({ url }) => workspacePath.test(url.pathname) ? '/' : undefined;
