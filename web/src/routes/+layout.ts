// The whole app runs client-side (the AI executes in a Web Worker in the browser),
// so there is nothing to server-render. Ship a prerendered static shell that
// hydrates into the SPA — ideal for static hosting on Cloudflare.
export const ssr = false;
export const prerender = true;
