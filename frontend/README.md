# Attendance V3 — Frontend

React + Vite single-page app for the Attendance V3 system. Talks to the
FastAPI backend via REST + WebSockets/SSE for live updates.

## Local development

Backend must be running first (see the top-level [README.md](../README.md)
or run `..\scripts\Start-LocalDev.ps1` from the repo root). Then:

```powershell
npm install
npm run dev
```

The Vite dev server starts on `http://localhost:5173` with `/api` and `/ws`
proxied to `http://localhost:8000` (see `vite.config.js`).

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Vite dev server with HMR on port 5173 |
| `npm run build` | Production build into `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | Run ESLint over the codebase |

## Stack

- **React 19** + **Vite 8** + **Tailwind CSS 4**
- **react-router-dom** for routing
- **axios** for HTTP, **lucide-react** for icons
- ESLint with `@eslint/js`, `eslint-plugin-react-hooks`, and `eslint-plugin-react-refresh`

## CI

PRs touching `frontend/**` automatically run ESLint and a Vite build via
`.github/workflows/ci.yml` (the `frontend` job). Both must pass before
merge. See [../CLAUDE.md](../CLAUDE.md) for the agent PR protocol.
