# Bulk Enrollment Importer — Slice 1 Design (offline batch folder/CSV ingest)

**Status:** design, read-only research phase. No repo files modified.
**Scope:** fixes/ROADMAP.md §2.5 item 1 — "Batch folder/CSV importer with quality gate + rejects
report (unblocks the whole pilot)". ROADMAP §2 calls bulk enrollment **P0 — gates the entire pilot**.
This slice is the ROADMAP §2.3 *Primary: offline batch folder ingest* path only. Kiosk mode,
coverage dashboard UI, consent plumbing, and progressive enrollment are explicitly out of scope.

---

## 0. Verified current-state facts (all checked in source)

| Fact | Source |
|---|---|
| Enrollment today is one-at-a-time browser upload: `POST /api/v1/students/{id}/enroll` | `backend/app/api/v1/students.py:215-400` |
| Quality gate `_resolve_enrollment_min_quality()` reads env `ATTENDANCE_ENROLLMENT_MIN_QUALITY` per call, default 0.5, range [0.0,1.0], malformed → RuntimeError (fail closed); gate triggers on strict `quality_score < min_quality` | `students.py:59-93`, `students.py:310` |
| `extract_enrollment_embedding(face_tensor, *, triton_client=None) -> tuple[np.ndarray, float]` runs YOLO detect → best-detection crop (falls back to **whole-image resize when zero detections — it does NOT raise**) → LVFace embed; returns `(512-D embedding, quality in [0,1])`; resolves client via `get_triton_client()` at point of use | `backend/app/services/pipeline/orchestrator.py:58-131` |
| Public pipeline surface = facade `app.services.pipeline_service` only (`extract_enrollment_embedding`, `process_inference_batch`, …). Callers must not import pipeline submodules | `backend/app/services/pipeline_service.py`, attendance-v3-context skill |
| Triton seam: `set_triton_client_override()` / `get_triton_client()` in `app.infrastructure.triton.client` (NOT legacy `app.worker.triton_client`); override checked every call; never cache client module-level | `backend/app/infrastructure/triton/client.py:309-333` |
| `StudentEmbedding`: UUID PK, `student_id` FK CASCADE, `embedding` Vector(512) NOT NULL, `pose_label` String(32) default 'front', `quality_score` Float [0,1] check, `is_active` bool, `created_at`. Index `(student_id, pose_label, is_active)` is **non-unique** — multiple active rows per pose are schema-legal | `backend/app/domain/models/student.py:86-133` |
| `TemplateAuditLog`: `action` String(32), `event_metadata` JSONB, `student_embedding_id` FK SET NULL; existing actions `"created"` / `"archived"`, metadata keys `{"source": "students_enroll_api_v1"}`, `{"reason": "pose_reenrollment"}` | `student.py:136-178`, `students.py:350-382` |
| API rotation semantics: archive all active embeddings for `(student_id, pose_label)` with `FOR UPDATE`, audit each `"archived"`, insert new active row, audit `"created"`, single commit | `students.py:336-392` |
| `Student` columns: `id` UUID PK, `user_id` UUID **UNIQUE** FK→users CASCADE, `student_number` String(32) **UNIQUE**, `program` NOT NULL, `enrollment_year` NOT NULL (2000–2100), `graduation_year?`, `date_of_birth?`, `is_active` (server_default true). API refuses enrollment for inactive students (409) | `student.py:32-83`, `students.py:237-241` |
| `StudentService.get_by_student_number()` exists (strip + select) | `backend/app/services/student_service.py:29-31` |
| Script convention: standalone script builds its **own** asyncpg engine inside the async fn from `ATTENDANCE_DATABASE_URL` (normalized `postgresql://` → `postgresql+asyncpg://`), one `asyncio.run()` in `main()`, dispose at end, exit 0/1 | `backend/scripts/seed_demo_data.py` |
| Worker pattern: sync wrapper → exactly ONE `asyncio.run()` over a private async helper; loop-bound singletons never cross loops; worker code must not use FastAPI-loop Redis singleton | `backend/app/worker/tasks.py:272-329`, skill |
| Test fakes: `FakeTritonGrpcClient` (deterministic YOLO box + LVFace outputs; liveness raw `[0.05, 2.5]` → decoded quality ≈ low; `zero_detections=True` yields empty detection set); injected via `set_triton_client_override(fake)` in conftest fixture | `backend/tests/_fakes.py`, `backend/tests/conftest.py:164-170` |
| Existing enrollment tests monkeypatch `app.api.v1.students.extract_enrollment_embedding` and use an `_EnvOverride` helper for the quality env | `backend/tests/test_smoke_enrollment_quality.py` |
| **No `template_version` column exists anywhere; no consent fields exist.** Both are ROADMAP §2.4 non-negotiables that need product/migration decisions (see Open Questions) | grep across backend/app |
| CI lint = exactly `ruff check --select=F backend`; pytest needs live PG(15432)+Redis+migrated schema; `filterwarnings=["error"]` | AGENTS.md |
| Protected paths for auto-merge: `backend/app/**`, `backend/tests/**` — this design touches both ⇒ PR must carry `do-not-merge` or expect manual merge | AGENTS.md |

---

## 1. CLI contract

New console script (no new package deps): `python -m scripts.import_enrollments` run from `backend/`,
or `python backend/scripts/import_enrollments.py` — same layout as `seed_demo_data.py`.

```
python -m scripts.import_enrollments \
  --source ./intake-2026              # required: root of batch folder
  [--manifest ./intake-2026/manifest.csv]   # optional: CSV mode (else folder-convention scan)
  --out ./import-out-2026-08-24       # required: reports + journal + summary (never inside --source)
  [--dry-run]                         # full pipeline incl. Triton, zero DB writes, no journal writes
  [--overwrite]                       # rotate existing active pose template (API semantics);
                                      # default: skip rows whose (student, pose) already has an active template
  [--default-pose front]              # pose when manifest omits pose_label / filename has none
  [--limit N]                         # process first N manifest rows (smoke testing)
  [--min-quality X]                   # CLI override; otherwise ATTENDANCE_ENROLLMENT_MIN_QUALITY; else 0.5
  [-v/--verbose]
```

Exit codes: `0` run completed (rejects allowed — they are data problems reported, not run failures);
`2` usage/config error (bad args, missing dirs, unreadable manifest header, bad `--min-quality`);
`3` infrastructure abort (Triton unreachable/repeatedly failing, DB down) — committed rows stay
committed, rerun resumes.

### 1a. Manifest CSV mode

Header (exact, UTF-8, comma): `student_number,image_path[,pose_label]`

- `student_number`: required, trimmed, max 32 chars (matches `Student.student_number` String(32)).
- `image_path`: relative to `--source` (absolute paths rejected unless they resolve under `--source`;
  symlink/traversal escape ⇒ `unreadable` reject — fail closed).
- `pose_label`: optional; defaults to `--default-pose` ("front"). Normalized with API-equivalent
  rules: strip, lowercase, non-blank, ≤ 32 chars; violation ⇒ `duplicate-ref`? No — validation
  failures of the *manifest line itself* (missing student_number, blank image_path, bad pose label)
  get their own row outcome `invalid-row` recorded in reasons.csv with a detail message, so the five
  image-derived codes stay clean. (See taxonomy note in §4.)
- Blank lines skipped; duplicate header detected as invalid-row; BOM tolerated.
- Encoding: utf-8-sig; `csv.DictReader` (stdlib only).

### 1b. Folder-convention mode (no `--manifest`)

Scan `--source` (non-recursive by default; `--recursive` optional later):

- `<student_number>.jpg|.jpeg|.png` (case-insensitive ext) ⇒ ref + pose from `--default-pose`.
- `<student_number>__<pose>.jpg` (double underscore) ⇒ explicit multi-pose variant, e.g.
  `2024CS101__left.jpg`. Single underscore deliberately avoided because real student numbers contain `-`
  and digits but may contain `_`… double-underscore is unambiguous.
- Any other file: if it has an image extension ⇒ `unknown-ref` reject (stem doesn't parse as a ref we
  can bind); else ignored silently (e.g., `.csv`, `.DS_Store`) and counted as `ignored_files`.

Both modes funnel into one internal row type: `(row_index, student_ref, image_path, pose_label_raw)`.
Folder mode is just a generated manifest — identical downstream code.

---

## 2. Per-row pipeline — REUSED vs NEW

Ordering matters: cheapest checks first, Triton last (it is the scarce resource), DB write atomic last.

```
resolve student → journal skip → conflict skip → decode image → embed (+detect gate)
   → quality gate → [dry-run stop] → transactional write → journal append
```

### REUSED (no behavioral change)

| Function | From | Use |
|---|---|---|
| `extract_enrollment_embedding(tensor)` | `app.services.pipeline_service` (facade — never a submodule) | detection + crop + LVFace embed + quality proxy, per row |
| `get_triton_client()` (called inside the orchestrator) | `app.infrastructure.triton` | seam preserved automatically; tests inject via `set_triton_client_override()` |
| `_resolve_enrollment_min_quality()` | `app.api.v1.students` | exact gate semantics incl. fail-closed RuntimeError on malformed env. See layering note below. |
| Gate comparison `quality_score < min_quality` (strict) + default 0.5 | ATT-029 semantics | identical accept/refuse boundary; boundary equality passes |
| Rotation + audit write shape (archive actives w/ `with_for_update`, audit `"archived"`/`"created"`, single commit) | `students.py:336-392` | mirrored by NEW writer using the same row shapes/actions so audit history stays uniform |
| Engine/session bootstrap pattern, URL normalization, exit-code style | `seed_demo_data.py` | script skeleton |
| `FakeTritonGrpcClient`, `set_triton_client_override` | `tests._fakes`, `app.infrastructure.triton` | integration tests |

Layering note on the quality-gate helper: it lives in the api layer today. The import **service**
must not import from `app.api.*` (services ← api direction only). Recommended minimal fix: move
`_resolve_enrollment_min_quality` (+ its two constants) into `app/services/enrollment_import_service.py`
or better a neutral `app/core/enrollment_policy.py`, and re-export from `app.api.v1.students`
(one import line) so `test_smoke_enrollment_quality.py` (imports `from app.api.v1.students import
_resolve_enrollment_min_quality`) stays green untouched. Fallback if that refactor is unwanted:
duplicate the ~20-line pure function in the service with a cross-reference comment and a unit test
asserting both copies agree on default/range/fail-closed. Relocation preferred; drift is the enemy.

### NEW

| Piece | Where | Notes |
|---|---|---|
| `main()` + argparse + exit codes | `backend/scripts/import_enrollments.py` (~120 lines) | thin shim: parse args → `asyncio.run(_run_import_async(args))` — exactly ONE `asyncio.run` per process, matching worker rule |
| `_run_import_async(args)` | service module | creates engine + session factory INSIDE the async fn (loop-bound lifetime confined to this loop), disposes in `finally`; never touches FastAPI-loop singletons or Redis |
| Manifest parser `parse_manifest(path)` / folder scanner `scan_folder(source, default_pose)` returning typed `ImportRow`s | service module | pure, DB-less, unit-testable |
| `_decode_image_file(path) -> np.ndarray` | service module | PIL `Image.open(path)` → RGB → float32 HWC contiguous; same checks as `_decode_enrollment_image` (≥8×8, ndim==3) but file-based; errors map to `unreadable` |
| Detection-presence gate | see below | orchestrator currently silently resizes the whole frame when YOLO finds nothing — that path must become a `no_face` reject instead of a garbage-template risk |
| `import_rows(rows, session_factory, opts) -> ImportSummary` | service module | the loop below |
| Journal reader/writer, rejects copier, reasons.csv writer, summary builder | service module | §3–§4 |
| Pose-label normalizer (file-local copy of API rules) | service module | 6 lines; avoids importing FastAPI machinery into the script |

### Detection-presence gate (the one orchestrator touch)

`extract_enrollment_embedding` returns an embedding even with zero detections (whole-image resize
fallback, `orchestrator.py:103-108`). Options considered:

1. Pre-run detection separately in the importer — requires `_frame_to_model_input`/`_parse_detections`
   from submodules ⇒ violates the facade contract. Rejected.
2. Accept fallback and rely on the quality gate to catch no-face images — unreliable: a textured
   wall photo can produce a mid-quality embedding. Rejected as primary.
3. **Chosen:** add a backward-compatible keyword to the orchestrator:

```python
class NoFaceDetectedError(ValueError): ...

async def extract_enrollment_embedding(face_tensor, *, triton_client=None,
                                       require_detection: bool = False):
    ...
    if not detections:
        if require_detection:
            raise NoFaceDetectedError("No face detected in enrollment image.")
        aligned_face = _resize_nearest(...)   # existing behavior unchanged
```

Importer always passes `require_detection=True` ⇒ maps to `no_face` reject. Default `False` keeps
the API route byte-for-byte behavior. `NoFaceDetectedError` subclasses `ValueError` (already in the
route's generic `except Exception` net) and is exported through the facade `__all__`.
~15 lines changed in `orchestrator.py` + 2 lines in `pipeline_service.py` + `pipeline/__init__.py`.

### Row state machine (normative)

| Step | Failure | Outcome |
|---|---|---|
| resolve `student_number` → active Student | not found OR `is_active=False` | `unknown_ref` reject (fail closed; detail distinguishes "inactive") |
| journal contains `(ref, sha256(image))` | hit | `skipped_resumed` |
| active template exists for `(student_id, pose_label)` and not `--overwrite` | hit | `skipped_active_exists` |
| decode image bytes | PIL/OSError/too-small/path-escape | `unreadable` reject |
| `extract_enrollment_embedding(..., require_detection=True)` | `NoFaceDetectedError` | `no_face` reject |
| ″ | TritonTimeoutError / ServerUnavailable / ModelUnavailable / InferenceError ×N retries (default 2 in-process retries, backoff 2s/8s) then give up | **abort run, exit 3** — infra failure is not a photo problem; do NOT write a reject row that tells staff to "re-shoot" |
| `quality_score < min_quality` | strict less-than, same boundary as API | `low_quality` reject |
| `--dry-run` | — | record would-import, continue (no DB write, no journal) |
| transaction: [archive+audit]* → insert StudentEmbedding → audit created → commit | IntegrityError/other | rollback ⇒ `db_error` counter (not a reject code), run continues; row simply not journaled so rerun retries it |
| journal append `(run_id, ref, sha256, pose, embedding_row_id)` + flush/fsync | — | crash here = row re-embeds next run (safe, idempotent) |

Transaction contents (mirrors API exactly): archived actives get
`TemplateAuditLog(action="archived", event_metadata={"reason": "bulk_import_rotation"})`; the new row
gets `action="created"`, `event_metadata={"source": "enrollment_import_cli", "run_id": run_id}`.
One commit per row ⇒ crash between embed and DB write loses nothing; DB never holds half a rotation.

---

## 3. Idempotency, resume, dry-run

- **Journal:** `--out/journal.jsonl`, one JSON object per committed row:
  `{"run_id": "...", "student_number": "...", "image_sha256": "...", "pose_label": "...",
  "embedding_id": "...", "committed_at": "..."}` — appended + flushed after each successful commit.
  On startup the importer loads it into a set keyed `(student_number, image_sha256)`. Re-running the
  same batch skips those rows entirely (no Triton call) ⇒ "re-running skips already-embedded
  students" (ROADMAP §2.3). `--overwrite` bypasses the *conflict* skip but still honors the journal
  (identical input bytes already imported ⇒ nothing to overwrite).
  Why content-hash keyed rather than "query DB for active templates": DB-truth conflates "imported by
  an importer run" with "enrolled earlier via API/kiosk"; the hash key gives exact resume semantics
  and works even across different target databases. The separate `skipped_active_exists` check covers
  the cross-channel case.
- **Crash safety:** embed→write is not checkpointed (worst case: one wasted inference on rerun);
  write→journal is ordered commit-first. A crash between them re-imports cleanly because insertion
  is additive (new UUID row); with `--overwrite` a rerun rotates again — idempotent end-state either way.
- **Dry-run:** executes everything up to and including Triton + quality gate, records
  `would_import` / would-reject outcomes, writes full report set, performs **zero** DB writes and
  zero journal writes. Purpose: validate a manifest against the real model before touching biometric
  storage.

---

## 4. Rejects + coverage reporting

Per rejected row: copy the original image to
`--out/rejects/<reason_code>/<student_number>__<original_name>` (sanitized; collisions suffixed
`__2`) and append to `--out/rejects.csv`:

```
run_id,student_number,image_path,pose_label,reason_code,detail,quality_score,rejected_at
```

Reason-code taxonomy (closed set, snake_case):

| Code | Meaning | Trigger |
|---|---|---|
| `unreadable` | not decodable as an image, <8px, or path escapes `--source` | decode step |
| `no_face` | detector found zero faces (`require_detection=True`) | orchestrator gate |
| `low_quality` | `quality_score < min_quality` (strict, same boundary as API) | ATT-029 gate |
| `duplicate_ref` | same `(student_number, pose_label)` appeared earlier **in this run's input** (manifest dup or folder-mode name collision after sanitize) | input validation, before any I/O on the second occurrence — first occurrence wins |
| `unknown_ref` | student_number absent from DB **or** student inactive (fail closed, detail says which) | identity resolution |

Manifest-line malformations (missing field, blank path, bad pose label) are logged to reasons.csv
with reason `invalid_row` — kept out of the five image/data codes above so ops metrics stay clean;
it is an input-authoring error, not a photo problem. DB-level per-row write failures are counted as
`db_errors` in the summary, not rejects (retryable by rerun).

End-of-run coverage line (stdout + `--out/import_summary.json`), ROADMAP §2.4 dashboard seed:

```
COVERAGE: enrolled 812/1000 active students (81.2%) | rows: total 1020 imported 790
skipped_resumed 22 skipped_active_exists 15 | rejected: unreadable=5 no_face=12 low_quality=48
duplicate_ref=3 unknown_ref=20 invalid_row=5 | db_errors=0 | run_id=<uuid4>
```

`enrolled N/M`: M = count of active Students (roster snapshot taken at start); N = distinct
active-template student_ids after the run (`SELECT COUNT(DISTINCT student_id) FROM student_embeddings
WHERE is_active`). `import_summary.json` additionally carries per-reason counts, duration, min_quality
in effect, and flags (dry_run/overwrite) — identifiers and counts only (see §5).

---

## 5. Privacy constraints (hard rules, enforced in code review + tests)

- Embeddings are biometric data of possibly-minors: the importer **never logs, prints, serializes,
  or writes** an embedding vector. No exceptions — including debug/verbose mode, tracebacks, and
  `repr()` of ORM objects (never log model instances; log ids only).
- reasons.csv / import_summary.json / journal.jsonl carry identifiers (`student_number`, UUIDs),
  reason codes, the scalar `quality_score` (same exposure as the existing API INFO log line),
  timestamps, and file paths. Nothing else.
- `TemplateAuditLog.event_metadata` gets `{"source": "enrollment_import_cli", "run_id": ...}` only.
- Rejects copies keep raw photos **inside the operator-supplied output tree** (local disk, operator
  owns retention); nothing is uploaded anywhere; the script makes no network calls except Triton + Postgres.
- Verbose logging mirrors API log discipline: ref, pose, outcome, quality scalar — never tensors.
- A unit test greps generated report files for 512-length float sequences / `embedding` keys to pin this.
- Never put embeddings in commits/PRs/test fixtures (repo-wide rule; fixtures use synthetic images).

---

## 6. Identity binding

Actual `Student` join candidates (verified columns): `id` (UUID PK), `user_id` (unique FK→users),
`student_number` (unique String(32)). External SIS exports carry human keys, not our UUIDs or our
auth-user ids; `email` lives on `User` and couples enrollment to account provisioning order.

**Recommendation: bind on `student_number`** — unique-constrained at the DB (`uq_students_student_number`),
SIS-native, matches ROADMAP's own filename example (`2024CS101.jpg`), used by seed data, and already
has a lookup helper (`StudentService.get_by_student_number`). Resolution loads one roster snapshot
(`SELECT id, student_number, is_active FROM students`) into a dict at start: O(1) per row, one query
total, and gives us M for the coverage denominator. Fail closed on missing/inactive ⇒ `unknown_ref`.
Optional future extension: accept an explicit `student_uuid` column when manifests are generated by
our own tooling; slice 1 does not need it.

Duplicate-ref policy (recommended): first occurrence in input order wins; subsequent
`(student_number, pose_label)` duplicates in the same run ⇒ `duplicate_ref` reject (prevents silent
last-one-wins mis-assignment — the failure mode ROADMAP §2.3 explicitly warns about). Cross-run
re-submission of the same file is handled by the journal (`skipped_resumed`), not treated as a dup.
Same image content bound to two *different* refs: flagged in summary under a warning counter
(suspected mass-upload error) but still imported — flagged as open question #6.

---

## 7. Triton seam & event-loop compliance

- Import Triton only via `app.infrastructure.triton`; the importer itself never constructs or caches
  a client — `extract_enrollment_embedding` resolves `get_triton_client()` at point of use, so
  `set_triton_client_override(fake)` governs tests exactly as for the API route. No module-level
  client, no default-arg caching.
- Script shape = worker-task shape: `def main(): asyncio.run(_run_import_async(args))` — one
  `asyncio.run` per process; engine + sessions created inside that coroutine and disposed before it
  returns; no second loop anywhere; no blocking calls inside the loop beyond PIL decode, which is
  acceptable in a standalone script (no FastAPI loop exists; optionally `asyncio.to_thread` for large
  images — noted, not required).
- No Redis involvement at all ⇒ the FastAPI-loop Redis singleton hazard doesn't apply; do not add one.
- Concurrency posture for slice 1: strictly sequential rows. Triton batching is a measured
  optimization deferred until throughput demands it (open question #8).

---

## 8. Test plan

Conventions honored: live Postgres (host port 15432) + Redis + migrated schema via fixtures that
already exist; `ATTENDANCE_TRITON_URL=fake-host:8001`; `filterwarnings=["error"]`; ruff F-only.

**A. DB-less unit tests** (`backend/tests/test_enrollment_import_unit.py`, no engine, no Triton):
1. Manifest parser: happy header; missing/blank fields → invalid_row; BOM; duplicate header; CRLF.
2. Path safety: absolute path outside source, `../` traversal, symlink escape → unreadable (fail closed).
3. Folder scanner: `<ref>.jpg`, `<ref>__<pose>.jpg`, case-insensitive extensions, ignored non-image
   files, unknown-ref stems.
4. Duplicate-ref policy: first-wins ordering, per-(ref,pose) granularity.
5. Reason-code mapping table exhaustively covered (every failure → exactly one code).
6. Journal: round-trip, skip-hit, corrupt-line tolerance (warn + ignore, don't abort resume).
7. Summary math + COVERAGE line format snapshot.
8. Privacy: build synthetic report files, assert no `embedding` key and no 512-float run (regex).
9. Pose normalization parity with API rules (blank/long/case).
10. If relocation chosen: `app.api.v1.students._resolve_enrollment_min_quality is
    app.core.enrollment_policy._resolve_enrollment_min_quality` plus existing ATT-029 helper tests stay green.

**B. Integration tests** (`backend/tests/test_enrollment_import_integration.py`, live DB +
`fake_triton` fixture / `set_triton_client_override`):
11. Happy path: seeded students + synthetic JPEGs → StudentEmbedding rows active with correct
    pose/quality, two TemplateAuditLog actions per rotated pose, metadata `source=enrollment_import_cli`.
12. Quality gate: `ATTENDANCE_ENROLLMENT_MIN_QUALITY=0.9` (via the ATT-029-style `_EnvOverride`)
    turns accepts into low_quality rejects; **zero** embedding/audit rows written for refused rows;
    boundary equality accepted at default 0.5. NOTE practical detail: `FakeTritonGrpcClient` decodes
    liveness `[0.05, 2.5]` to a LOW quality (~sigmoid ≈ 0.07), so accept-paths must either set env
    threshold to 0.0 or monkeypatch the service-level extractor — mirror the existing ATT-029 test
    approach and say so in fixtures.
13. `require_detection=True`: fake with `zero_detections=True` → no_face reject, no rows.
14. Idempotent resume: run twice → second run all `skipped_resumed`, no new embedding rows, Triton
    call-count unchanged (assert via `fake.calls`).
15. `--overwrite`: second run rotates (old active → inactive + `"archived"` audit; new row + `"created"`).
16. Dry-run: report written, DB shows zero new embeddings, journal file absent/empty.
17. Rejects: unreadable bytes file, unknown ref, inactive student, duplicate ref → files land in
    `rejects/<code>/`, reasons.csv rows correct, images copied byte-identical.
18. Abort path: fake raising `TritonServerUnavailableError` → exit code 3, prior commits intact.
19. Crash-injection (light): kill between commit and journal via a patched journal writer raising —
    rerun re-imports that row exactly once more (no duplicates beyond the intended retry).

Acceptance criteria ↔ ROADMAP non-negotiables:

| ROADMAP requirement | Covered by |
|---|---|
| §2.3 primary: folder-or-CSV ingest named by identifier | §1a/§1b, tests 1–4 |
| §2.3: detect → quality-gate → embed → upsert → per-row result report | §2 flow + rejects/report §4 |
| §2.3: resumable + idempotent, skips already-embedded | §3 journal, tests 14–15 |
| §2.3: rejects folder with reasons for re-shooting | §4, tests 17 |
| §2.4: quality gate BEFORE storing any template | reused ATT-029 gate pre-insert, test 12 |
| §2.4: multiple embeddings per student | many-to-one schema + per-pose rows (multi-pose manifests) |
| §2.4: template_version stored | **NOT in slice 1** — column doesn't exist; migration decision needed (OQ#3) |
| §2.4: consent capture at enrollment | partial: audit metadata marks provenance; real consent plumbing is ROADMAP slice 4 (OQ#4) |
| §2.4: deletion path incl. derived templates | already satisfied by FK CASCADE `students→student_embeddings`/audit; importer adds nothing to delete |
| §2.4: coverage "N of M enrolled, K low-quality, J failed" | COVERAGE line + import_summary.json (dashboard UI = slice 2) |
| Skill: facade / Triton seam / one-loop / privacy rules | §7 + tests 10, 14 (call-count), privacy test 8 |

Files-touched estimate:

| File | Status | Est. size |
|---|---|---|
| `backend/scripts/import_enrollments.py` | NEW | ~120 lines |
| `backend/app/services/enrollment_import_service.py` | NEW | ~400 lines (parser/scanner/pipeline/journal/reports) |
| `backend/app/services/pipeline/orchestrator.py` | MODIFY | +~15 (`NoFaceDetectedError`, `require_detection`) — protected path |
| `backend/app/services/pipeline/__init__.py` + `pipeline_service.py` | MODIFY | +2 export lines each |
| `backend/app/core/enrollment_policy.py` (if relocation chosen) | NEW | ~40 moved lines |
| `backend/app/api/v1/students.py` (if relocation chosen) | MODIFY | −20/+2 (re-export) — protected path |
| `backend/tests/test_enrollment_import_unit.py` | NEW | ~300 lines — protected path |
| `backend/tests/test_enrollment_import_integration.py` | NEW | ~350 lines — protected path |
| Migrations / frontend / .env.example | none | reuse `ATTENDANCE_ENROLLMENT_MIN_QUALITY`; consider documenting it (OQ#9) |

All backend/** changes ⇒ auto-merge workflow will not squash these automatically even with `agent-pr`
label (protected paths) — plan for manual merge; still apply `do-not-merge` if review is desired.

---

## 9. Open questions (need product/owner decisions)

1. **Authoritative identifier:** is `student_number` truly the stable SIS key (survives re-enrollment
   years, no format collisions across faculties)? Or should manifests carry national/email/email-less
   registrar IDs mapped through a future lookup?
2. **Overwrite policy default:** proposed default = skip if an active template already exists for
   (student, pose), with opt-in `--overwrite` rotating API-style. Alternative: always rotate (API
   parity) or allow multiple simultaneous active templates per pose (schema permits; matching currently
   assumes...?). Which invariant should the system standardize on — one-active-per-pose or many?
3. **`template_version`:** ROADMAP §2.4 wants a stored version to re-embed populations on model change.
   Add the column now (one Alembic migration, cheap while we're here) or defer to a later slice?
   Downgrade must round-trip if added.
4. **Consent:** record a consent reference in `TemplateAuditLog.event_metadata` now (string passthrough
   from manifest?) or leave consent wholly to slice 4 plumbing?
5. **Inactive students:** fold into `unknown_ref` (proposed, fail closed) or dedicated reason code so
   coverage dashboards can distinguish "bad roster" from "photo problem"?
6. **Identical image bound to multiple refs** (mass-upload mistake signature): reject later ones as
   duplicate_ref, or import + warn-counter (proposed)?
7. **Quality-helper relocation** out of the api layer (small protected-path refactor + re-export) vs
   duplicated local copy — confirm appetite for touching `students.py` again.
8. **Throughput target:** sequential-per-row is simplest and safest; if >~5k images/run matters,
   spec Triton batch size and parallel decode before implementation freezes.
9. Should `ATTENDANCE_ENROLLMENT_MIN_QUALITY` be documented in `.env.example` as part of this slice?
