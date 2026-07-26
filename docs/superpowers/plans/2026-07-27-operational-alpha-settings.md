# Operational Alpha Settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, audited alpha preset workflow to the authenticated Vercel dashboard.

**Architecture:** A pure TypeScript policy module owns alpha validation and retrospective preset metadata. An authenticated App Router API reads settings and invokes one atomic Supabase RPC for writes. A client settings page guides the user through selection, impact review, reason entry, confirmation, and pending-state feedback.

**Tech Stack:** Next.js 16 App Router, React 19, TypeScript, Supabase Postgres/Auth/RLS, Vitest, Lucide React.

## Global Constraints

- Allowed alpha values are exactly `0.005` and `0.010`.
- Alpha changes confidence display only; autonomous alert membership is unchanged.
- TEST metrics are retrospective reference values and do not automatically select a preset.
- Saving a request does not change the detector's effective alpha.
- Only authenticated members with `can_write=true` may request changes.
- Every change is append-only audited.

---

### Task 1: Alpha Policy Contract

**Files:**
- Create: `vercel-dashboard/src/lib/alpha-settings.ts`
- Create: `vercel-dashboard/tests/alpha-settings.test.ts`

**Interfaces:**
- Produces: `ALPHA_PRESETS`, `parseAlphaRequest`, and alpha setting response types.

- [ ] Write failing tests for allowed presets, arbitrary alpha rejection, reason length, and confirmation.
- [ ] Run the focused test and verify the missing module failure.
- [ ] Implement the minimal pure validation module.
- [ ] Run the focused test and verify pass.

### Task 2: Supabase Persistence

**Files:**
- Create: `vercel-dashboard/supabase/migrations/202607270001_operational_alpha_settings.sql`
- Create: `vercel-dashboard/tests/alpha-settings-migration.test.ts`

**Interfaces:**
- Produces: `operational_alpha_settings`, `operational_alpha_setting_events`, and `request_operational_alpha_change`.

- [ ] Write a failing migration contract test.
- [ ] Add tables, constraints, RLS read policies, grants, and atomic writer RPC.
- [ ] Run the migration contract test.

### Task 3: Authenticated API

**Files:**
- Create: `vercel-dashboard/app/api/settings/alpha/route.ts`
- Create: `vercel-dashboard/tests/alpha-settings-route.test.ts`

**Interfaces:**
- Consumes: authenticated Supabase client and Task 1 parser.
- Produces: GET state/history and POST pending request.

- [ ] Write failing tests for authentication, validation, read response, and RPC mutation.
- [ ] Implement GET and POST route handlers.
- [ ] Run focused route tests.

### Task 4: Settings UI

**Files:**
- Create: `vercel-dashboard/app/settings/alpha/page.tsx`
- Create: `vercel-dashboard/app/ui/alpha-settings.tsx`
- Modify: `vercel-dashboard/app/ui/dashboard.tsx`
- Modify: `vercel-dashboard/app/globals.css`
- Create: `vercel-dashboard/tests/alpha-settings-ui.test.ts`

**Interfaces:**
- Consumes: `/api/settings/alpha`.
- Produces: authenticated preset workflow and dashboard settings navigation.

- [ ] Write failing structural UI tests for presets, impact warning, reason, confirmation, and settings navigation.
- [ ] Implement the responsive settings page using the existing visual system and Lucide icons.
- [ ] Run focused UI tests.

### Task 5: Verification

**Files:**
- Modify: `vercel-dashboard/README.md`
- Modify: `vercel-dashboard/.env.example` only if required by the implementation.

- [x] Run `npm test`.
- [x] Run `npm run build`.
- [x] Start the local dev server on an unused port.
- [x] Verify desktop and mobile layouts with Playwright screenshots.
- [x] Confirm the existing dashboard and Shadow routes still build and remain covered by the full test suite.

## Phase 2: Mac Application Acknowledgement

- [x] Add a version-aware acknowledgement RPC and immutable audit event.
- [x] Add an atomic local Alpha configuration writer and strict loader.
- [x] Reuse the authenticated dashboard exporter session for polling.
- [x] Keep dashboard status export alive when Alpha synchronization fails.
- [x] Refresh desired/effective application state in the UI.
- [ ] Apply both Alpha migrations to the remote Supabase project.
- [ ] Deploy the committed dashboard branch and verify the live workflow.
