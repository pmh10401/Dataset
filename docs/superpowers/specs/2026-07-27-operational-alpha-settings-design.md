# Operational Alpha Settings Design

## Goal

Add a safe alpha-setting workflow to the authenticated Vercel service
dashboard. Alpha changes only which existing autonomous Hard alerts receive a
strong-confidence display label; they do not create, remove, or move alerts.

## User Flow

The dashboard header links to `/settings/alpha`. The settings page shows the
effective alpha reported by the detector, the requested alpha stored by the
service, and the synchronization status. A user chooses one of two frozen
presets:

- `0.005` Strict: retrospective strong-confidence count 228 and precision
  92.54%.
- `0.010` Balanced: retrospective strong-confidence count 361 and precision
  91.41%.

The page explicitly states that autonomous Hard alert count remains unchanged
and that benchmark figures are retrospective reference values, not production
guarantees. The user must enter a reason and confirm this behavior before
submitting.

## State Model

`desired_alpha` is the user's requested setting. `effective_alpha` is the value
last acknowledged by the detector. The service displays `pending` until a
future detector synchronization phase confirms application. Saving a request
must never change `effective_alpha`.

Every request increments a version and appends an immutable event containing
the prior desired value, requested value, authenticated user, reason, and
timestamp.

## Authorization

All authenticated dashboard members may read settings and history. Only a
member with `dashboard_members.can_write = true` may request a change. The
database mutation is exposed through a `security definer` Postgres function
that checks `auth.uid()` and membership before atomically updating current
state and appending history.

## API

- `GET /api/settings/alpha`: authenticated read of current setting, recent
  history, write permission, preset metadata, and retrospective impact.
- `POST /api/settings/alpha`: authenticated request using only alpha `0.005` or
  `0.010`, a reason of 5-500 trimmed characters, and an explicit confirmation.

The API does not accept arbitrary alpha values and does not claim detector
application.

## Scope Boundary

This first phase implements the settings UI, authenticated persistence, audit
history, and truthful pending status. Detector polling and applied-status
acknowledgement are a separate second phase.
