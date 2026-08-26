// Pure decision + pose-guidance logic for the live enrollment UX.
// Extracted from Enrollment.jsx so the state machines stay unit-testable
// without jsdom, canvas, or a webcam (mirrors recognizeErrors.js precedent).

export const PREVIEW_ENDPOINT = '/api/v1/students/enroll/preview'
export const PREVIEW_INTERVAL_MS = 600
export const PREVIEW_MAX_SIDE = 640

export const POSE_SEQUENCE = ['Look straight', 'Turn slightly left', 'Turn slightly right']
export const POSE_LABELS = ['front', 'left', 'right']
export const REQUIRED_CONSECUTIVE_OK = 2

// Errors that must abort the guided capture entirely (surfaced verbatim);
// everything else (e.g. 422 quality refusal) fails one pose without stopping.
export const HARD_ERROR_STATUSES = [401, 403, 413, 503]

// A preview is actionable only when a face was detected AND every server-side
// diagnostic passed (reasons == []). This is the exact contract of
// POST /students/enroll/preview: ok === detected && reasons.length === 0,
// recomputed client-side so a stale/malformed payload can never advance the
// pose machine.
export function decideOk(preview) {
  return (
    Boolean(preview?.detected) &&
    Array.isArray(preview?.reasons) &&
    preview.reasons.length === 0
  )
}

export function initialPoseMachine() {
  return { poseIndex: 0, consecutiveOk: 0, completedCount: 0 }
}

export function poseDone(machine) {
  return machine.completedCount >= POSE_SEQUENCE.length
}

// One preview evaluation fed into the guidance machine. Two consecutive ok
// previews complete the current pose ("hold the pose steady"), any non-ok
// preview resets the streak. Completing the final pose finishes the sequence.
export function advancePose(machine, isOk) {
  if (poseDone(machine)) return machine

  if (!isOk) {
    if (machine.consecutiveOk === 0) return machine
    return { ...machine, consecutiveOk: 0 }
  }

  const consecutiveOk = machine.consecutiveOk + 1
  if (consecutiveOk < REQUIRED_CONSECUTIVE_OK) {
    return { ...machine, consecutiveOk }
  }

  const completedCount = machine.completedCount + 1
  return {
    poseIndex: Math.min(completedCount, POSE_SEQUENCE.length - 1),
    consecutiveOk: 0,
    completedCount,
  }
}

export function isHardError(err) {
  return HARD_ERROR_STATUSES.includes(err?.response?.status)
}
