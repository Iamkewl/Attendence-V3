# Attendance V3 — agent collaboration rules

Project-scope instructions for any AI agent (Claude, Copilot CLI, future ones)
working in this repo. Loaded automatically by Claude Code; Copilot CLI sessions
should be primed by including the relevant section in the task brief.

---

## Branching and PR protocol

The `main` branch is protected:

- Direct pushes are blocked.
- A PR is required for every change.
- `backend-tests / smoke` must be green before merge.
- Squash-merge only; linear history enforced.
- Branches are auto-deleted on merge.

### Opening an agent PR

1. Branch from up-to-date `main`. Naming: `agent/<short-slug>` for autonomous
   work, `fix/<slug>` or `feat/<slug>` for human-initiated work.
2. Make the change. Run `pytest -v` locally before pushing.
3. Open the PR with `gh pr create`.
4. Apply the `agent-pr` label. This is what enables the auto-merge workflow:
   ```
   gh pr edit <num> --add-label agent-pr
   ```
5. If anything looks risky, also apply `do-not-merge` — this overrides the
   `agent-pr` label and forces manual review.

### Shortcut: one command does steps 1–5

The user-scope script `New-AgentPR.ps1` (at `C:\Users\DELL\.agents\scripts\`)
packages an entire dirty working tree into a labeled agent PR. From the
repo root with uncommitted changes:

```powershell
powershell -File C:\Users\DELL\.agents\scripts\New-AgentPR.ps1 `
    -Slug fix-something -Title "area: short imperative subject" `
    -AgentName Claude -ModelName "Claude Opus 4.7"
```

It cuts `agent/fix-something` from current state, commits with the right
Co-Authored-By trailer, pushes, opens the PR, and applies the `agent-pr`
label. Add `-DryRun` to see the plan without touching git. Add `-NoLabel`
when the PR is expected to need manual review (e.g. touches CI files).

### What auto-merge will refuse

The `.github/workflows/auto-merge.yml` guard refuses to enable auto-merge if
the PR diff touches **any** of these paths:

- `.github/**` (CI workflows, CODEOWNERS, issue templates)
- `backend/alembic/**`, `backend/alembic.ini` (DB migrations)
- `pyproject.toml`, `frontend/package*.json` (dependency manifests)
- `scripts/**`, `infra/**` (deploy and infra)
- `docker-compose.dev.yml`

For changes to these paths, the PR will sit waiting for Suryaansh to merge
manually. This is intentional — these paths are CODEOWNERS-protected and
also represent the attack surface for supply-chain or CI-tampering risks.

### What the auto-merger CANNOT do

The GitHub App backing the auto-merge workflow has `Workflows: No access`.
It is physically incapable of merging a PR that modifies `.github/workflows/`
even if the guard above had a bug. Defense in depth.

The App also lacks permission to push directly to `main` or to bypass branch
protection. It can only flip the auto-merge flag on PRs.

---

## Commit and PR message style

Follow the existing repo style (see recent `git log`):

- Short imperative subject under 70 chars, prefixed with area (`ci:`, `backend:`, `docs:`, etc.)
- Body explains *why*, not *what* — the diff shows what.
- Always include `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`
  (or the equivalent for whichever agent authored the change).

PR body should include:
- A `## Summary` section (1–3 bullets, what changed)
- A `## Test plan` section (what was verified, even if just "smoke suite passed")

---

## Testing

The backend smoke suite lives at `backend/tests/test_smoke_*.py` and requires
Postgres + Redis running locally. Triton is faked via
`FakeTritonGrpcClient` — no GPU needed.

```powershell
docker compose -f .\docker-compose.dev.yml up -d postgres redis
$env:ATTENDANCE_DATABASE_URL = "postgresql+asyncpg://attendance:attendance@localhost:15432/attendance"
$env:ATTENDANCE_DATABASE_URL_TEST = $env:ATTENDANCE_DATABASE_URL
$env:ATTENDANCE_REDIS_URL = "redis://localhost:6379/0"
$env:ATTENDANCE_JWT_SECRET = "test-secret-32chars-minimum-needed"
$env:ATTENDANCE_ALLOWED_ORIGINS = "http://localhost:3000"
$env:ATTENDANCE_TRITON_URL = "fake-host:8001"
python -m pytest
```

Expected: `8 passed` in roughly 5–10 seconds. If anything fails, fix it
before opening a PR — CI is the gate, not the discovery mechanism.

---

## Nightly inference regression (Kaggle)

The production GPU server is the source of truth for inference. As a
backup safety net for when it's offline, `.github/workflows/nightly-inference.yml`
runs the YOLOv12 + LVFace models on a Kaggle T4 daily at 04:00 UTC and
compares outputs to `tests/inference_baseline.json` within tolerance.

- Kernel script: `notebooks/nightly_inference.py`
- Models live in a private Kaggle Dataset (`<user>/attendence-v3-models`),
  seeded once via `infra/kaggle_dataset/`. See its README for upload steps.
- Baseline JSON is synced from the repo on every workflow run via the
  small `<user>/attendence-v3-baselines` dataset.
- Drift detection layers:
  1. Model file sha256 changed → loud fail.
  2. Output tensor sha256 mismatch → check stats drift.
  3. Stats drift > `1e-4` → fail. Otherwise PASS (acceptable CUDA noise).

When a scheduled run reports FAIL, the workflow opens an issue labeled
`nightly-drift` with the run URL and triage steps. Manual trigger via
`gh workflow run nightly-inference.yml`.

Iterate on the kernel locally with `scripts/Push-KaggleKernel.ps1 -Watch`
(needs `KAGGLE_USERNAME` env var and `~/.kaggle/kaggle.json` set up).

---

## What NOT to do

- Do not push directly to `main`. (You can't — branch protection rejects it.)
- Do not edit `.github/**` in a PR you expect to auto-merge. Use a separate
  human-reviewed PR.
- Do not commit secrets. The `.env` files and `*.pem` keys are in `.gitignore`
  but the burden is on you not to add new secret-bearing files.
- Do not commit model artifacts (`*.onnx`, `*.pt`, etc.) — they are in
  `.gitignore` and exceed GitHub's 100 MB per-file limit anyway.
- Do not run `git push --force` against shared branches.
- Do not amend or rewrite history of commits that have been pushed.
