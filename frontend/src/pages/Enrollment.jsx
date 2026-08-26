import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, Check, CheckCircle, RefreshCcw, UserCheck, UserPlus } from 'lucide-react'
import client from '../api/client'
import { useAuth } from '../auth/AuthContext'
import WebcamCapture from './WebcamCapture'
import StudentPicker from './StudentPicker'
import NewStudentModal from '../components/NewStudentModal'
import {
  POSE_LABELS,
  POSE_SEQUENCE,
  PREVIEW_ENDPOINT,
  PREVIEW_INTERVAL_MS,
  PREVIEW_MAX_SIDE,
  advancePose,
  decideOk,
  initialPoseMachine,
  isHardError,
  poseDone,
} from './enrollLogic'

const CREATE_STUDENT_ROLES = ['admin', 'instructor']

const ENROLL_ENDPOINT = (studentId) => `/api/v1/students/${studentId}/enroll`
const COVERAGE_URL = '/api/v1/admin/enrollment-coverage'
const QUALITY_WARNING_THRESHOLD = 0.6

function resolveErrorMessage(err) {
  const statusCode = err?.response?.status
  const detail = err?.response?.data?.detail

  if (statusCode === 400) {
    return detail || 'The image was rejected (bad format or too small to process).'
  }
  if (statusCode === 401) {
    return 'Your session has expired. Please sign in again.'
  }
  if (statusCode === 403) {
    return 'Insufficient role: enrolling faces requires instructor or admin privileges.'
  }
  if (statusCode === 413) {
    return 'Capture too large. Please retake with a smaller image (max 10 MB).'
  }
  if (statusCode === 404) {
    return 'Student not found. Please re-select a valid student.'
  }
  if (statusCode === 422) {
    return (
      detail ||
      'Image quality is too low. Please retake with better lighting and a clear, forward-facing view.'
    )
  }
  if (statusCode === 503) {
    return 'The face-embedding service is temporarily unavailable. Please try again in a moment.'
  }
  if (statusCode) {
    return (
      detail ||
      `Enrollment failed with HTTP status ${statusCode}. Please try again or contact an administrator.`
    )
  }
  return err.message || 'Enrollment failed due to a network error. Check your connection and retry.'
}

// ── Spinner ────────────────────────────────────────────────────────────────

function FullPageSpinner() {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '60px 0',
        gap: 16,
      }}
      role="status"
      aria-live="polite"
    >
      <div
        aria-label="Submitting enrollment"
        style={{
          width: 44,
          height: 44,
          border: '4px solid rgba(231,134,53,0.2)',
          borderTopColor: 'var(--brand)',
          borderRadius: '50%',
          animation: 'spin 0.75s linear infinite',
        }}
      />
      <p style={{ color: 'var(--text-soft)', fontWeight: 600, margin: 0 }}>
        Extracting face embedding...
      </p>
      {/* Keyframe injected inline — safe in JSX, doesn't touch any CSS file */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

// ── Success card ───────────────────────────────────────────────────────────

function SuccessCard({ record, qualityWarning, onEnrollAnother }) {
  const qualityPct =
    typeof record.quality_score === 'number'
      ? `${Math.round(record.quality_score * 100)}%`
      : 'N/A'

  const qualityColor =
    typeof record.quality_score === 'number' && record.quality_score >= QUALITY_WARNING_THRESHOLD
      ? 'var(--present-text)'
      : 'var(--late-text)'

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 18,
        padding: '28px 0 10px',
        textAlign: 'center',
      }}
      role="status"
      aria-live="polite"
    >
      <CheckCircle size={44} style={{ color: 'var(--present-text)' }} aria-hidden="true" />

      <h3
        style={{
          fontFamily: 'Space Grotesk, sans-serif',
          fontSize: '1.35rem',
          fontWeight: 700,
          color: 'var(--text-main)',
          margin: 0,
        }}
      >
        Enrollment Successful
      </h3>

      {qualityWarning && (
        <div className="alert-banner" role="alert" style={{ width: '100%', maxWidth: 520 }}>
          <AlertCircle size={16} aria-hidden="true" />
          <span>
            Quality score is below 60% — consider re-enrolling with better lighting or a clearer
            face view for more reliable recognition.
          </span>
        </div>
      )}

      {/* Result detail table */}
      <div
        style={{
          width: '100%',
          maxWidth: 520,
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          background: 'var(--surface-strong)',
          overflow: 'hidden',
          textAlign: 'left',
        }}
      >
        {[
          { label: 'Student', value: record.studentName || record.student_id },
          { label: 'Pose', value: record.pose_label, capitalize: true },
          { label: 'Quality Score', value: qualityPct, color: qualityColor, bold: true },
          { label: 'Template ID', value: record.id, mono: true },
          {
            label: 'Created',
            value: record.created_at ? new Date(record.created_at).toLocaleString() : 'N/A',
          },
        ].map(({ label, value, capitalize, color, bold, mono }) => (
          <div
            key={label}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              gap: 12,
              padding: '9px 14px',
              borderBottom: '1px solid rgba(15,23,42,0.07)',
            }}
          >
            <span style={{ fontSize: '0.84rem', color: 'var(--text-subtle)', whiteSpace: 'nowrap' }}>
              {label}
            </span>
            <span
              style={{
                fontSize: '0.92rem',
                color: color || 'var(--text-main)',
                fontWeight: bold ? 700 : 500,
                textTransform: capitalize ? 'capitalize' : undefined,
                fontFamily: mono ? 'monospace' : undefined,
                wordBreak: 'break-all',
                textAlign: 'right',
              }}
            >
              {value}
            </span>
          </div>
        ))}
      </div>

      <button type="button" className="solid-btn" onClick={onEnrollAnother}>
        <RefreshCcw size={15} aria-hidden="true" />
        <span>Enroll Another</span>
      </button>
    </div>
  )
}

// ── Live guided capture (phone-style) ─────────────────────────────────────

const chipStyle = (tone) => ({
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '4px 10px',
  borderRadius: 999,
  fontSize: '0.78rem',
  fontWeight: 600,
  border: `1px solid ${tone === 'ok' ? 'rgba(18,102,70,0.35)' : 'rgba(180,83,9,0.35)'}`,
  background: tone === 'ok' ? 'var(--present-bg)' : 'var(--late-bg)',
  color: tone === 'ok' ? 'var(--present-text)' : 'var(--late-text)',
})

// Draws the bbox overlay rectangle for one preview evaluation. The reason
// chips themselves are DOM elements (accessible + testable); only the
// rectangle goes on the canvas.
function drawBboxOverlay(canvasEl, data) {
  if (!canvasEl || typeof canvasEl.getContext !== 'function') return
  const ctx = canvasEl.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height)
  if (!Array.isArray(data?.bbox) || data.bbox.length !== 4) return
  const [x, y, w, h] = data.bbox
  ctx.strokeStyle = data.ok ? '#16a34a' : '#d97706'
  ctx.lineWidth = 3
  ctx.strokeRect(x * canvasEl.width, y * canvasEl.height, w * canvasEl.width, h * canvasEl.height)
}

function canvasToBlob(canvas) {
  return new Promise((resolve) => {
    if (!canvas || typeof canvas.toBlob !== 'function') {
      resolve(null)
      return
    }
    canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.92)
  })
}

function LiveEnrollmentPanel({ studentId }) {
  const videoRef = useRef(null)
  const workCanvasRef = useRef(null)
  const fullCanvasRef = useRef(null)
  const overlayCanvasRef = useRef(null)
  const streamRef = useRef(null)
  const tickRef = useRef(null)
  const pauseRef = useRef(false)
  const haltedRef = useRef(false)
  const poseRef = useRef(initialPoseMachine())
  const queueRef = useRef([null, null, null])
  const grabPromisesRef = useRef([null, null, null])

  const [cameraOn, setCameraOn] = useState(false)
  const [liveError, setLiveError] = useState('')
  const [previewChips, setPreviewChips] = useState([])
  const [previewOk, setPreviewOk] = useState(false)
  const [poseUi, setPoseUi] = useState(initialPoseMachine())
  const [capturedPoses, setCapturedPoses] = useState([false, false, false])
  const [poseOutcomes, setPoseOutcomes] = useState(null)
  const [submittingLive, setSubmittingLive] = useState(false)

  // Full-res frame grab for auto-capture at the moment a pose completes.
  const grabFullRes = useCallback(async () => {
    const video = videoRef.current
    const canvas = fullCanvasRef.current
    if (!video || !canvas || !video.videoWidth) return null
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    return canvasToBlob(canvas)
  }, [])

  const runSubmissions = useCallback(async () => {
    pauseRef.current = true
    setSubmittingLive(true)
    const outcomes = []
    for (let i = 0; i < POSE_SEQUENCE.length; i += 1) {
      const blob = await grabPromisesRef.current[i]
      if (!blob) {
        outcomes.push({ pose: POSE_SEQUENCE[i], status: 'failed', message: 'Frame missing — redo guided capture.' })
        continue
      }
      const formData = new FormData()
      formData.append('image_file', blob, `${POSE_LABELS[i]}.jpg`)
      formData.append('pose_label', POSE_LABELS[i])
      try {
        await client.post(ENROLL_ENDPOINT(studentId), formData, {
          headers: { 'Content-Type': undefined },
        })
        outcomes.push({ pose: POSE_SEQUENCE[i], status: 'saved' })
      } catch (err) {
        if (isHardError(err)) {
          haltedRef.current = true
          setLiveError(resolveErrorMessage(err))
          break
        }
        outcomes.push({ pose: POSE_SEQUENCE[i], status: 'failed', message: resolveErrorMessage(err) })
      }
    }
    setPoseOutcomes(outcomes)
    setSubmittingLive(false)
    pauseRef.current = false
  }, [studentId])

  const applyPreview = useCallback(
    (data) => {
      drawBboxOverlay(overlayCanvasRef.current, data)
      const ok = decideOk(data)
      setPreviewOk(ok)
      setPreviewChips(Array.isArray(data?.reasons) ? data.reasons : [])

      if (pauseRef.current || haltedRef.current || poseDone(poseRef.current)) return

      const prev = poseRef.current
      const next = advancePose(prev, ok)
      if (next === prev) return
      poseRef.current = next
      setPoseUi(next)

      if (next.completedCount > prev.completedCount) {
        const poseIdx = prev.poseIndex
        grabPromisesRef.current[poseIdx] = grabFullRes().then((blob) => {
          queueRef.current[poseIdx] = blob
          setCapturedPoses((q) => q.map((v, idx) => (idx === poseIdx ? true : v)))
          return blob
        })
        if (next.completedCount === POSE_SEQUENCE.length) {
          void runSubmissions()
        }
      }
    },
    [grabFullRes, runSubmissions],
  )

  // One sampler tick: downscale → JPEG → preview POST → overlay + guidance.
  const runTick = useCallback(async () => {
    if (pauseRef.current) return
    const video = videoRef.current
    const work = workCanvasRef.current
    if (!video || !work || !video.videoWidth) return

    const scale = PREVIEW_MAX_SIDE / Math.max(video.videoWidth, video.videoHeight)
    work.width = Math.round(video.videoWidth * scale)
    work.height = Math.round(video.videoHeight * scale)
    const overlay = overlayCanvasRef.current
    if (overlay) {
      overlay.width = video.videoWidth
      overlay.height = video.videoHeight
    }
    const ctx = typeof work.getContext === 'function' ? work.getContext('2d') : null
    if (!ctx) return
    ctx.drawImage(video, 0, 0, work.width, work.height)

    const blob = await canvasToBlob(work)
    if (!blob) return

    const formData = new FormData()
    formData.append('image_file', blob, 'preview.jpg')
    try {
      const response = await client.post(PREVIEW_ENDPOINT, formData, {
        headers: { 'Content-Type': undefined },
      })
      applyPreview(response.data)
    } catch (err) {
      if (isHardError(err)) {
        haltedRef.current = true
        setLiveError(resolveErrorMessage(err))
      }
    }
  }, [applyPreview])

  useEffect(() => {
    tickRef.current = runTick
  })

  // Sampler lifecycle — cleared on unmount or camera stop.
  useEffect(() => {
    if (!cameraOn) return undefined
    const id = window.setInterval(() => {
      void tickRef.current?.()
    }, PREVIEW_INTERVAL_MS)
    return () => window.clearInterval(id)
  }, [cameraOn])

  const handleEnableCamera = useCallback(async () => {
    setLiveError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      })
      streamRef.current = stream
      setCameraOn(true)
    } catch (err) {
      const name = err?.name || ''
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setLiveError('Camera permission was denied. Please allow camera access and try again.')
      } else if (name === 'NotFoundError') {
        setLiveError('No camera detected on this device.')
      } else {
        setLiveError(err.message || 'Unable to access camera.')
      }
    }
  }, [])

  // Attach stream once the <video> mounts; stop tracks on unmount/teardown
  // (same hygiene as WebcamCapture's manual path).
  useEffect(() => {
    if (!cameraOn) return undefined
    const video = videoRef.current
    if (video && streamRef.current) {
      video.srcObject = streamRef.current
      try {
        const played = video.play()
        played?.catch?.(() => {})
      } catch {
        /* autoplay rejection is fine — user gesture already happened */
      }
    }
    return () => {
      streamRef.current?.getTracks?.().forEach((track) => track.stop())
      streamRef.current = null
    }
  }, [cameraOn])

  const handleResetGuided = useCallback(() => {
    poseRef.current = initialPoseMachine()
    queueRef.current = [null, null, null]
    grabPromisesRef.current = [null, null, null]
    haltedRef.current = false
    setPoseUi(initialPoseMachine())
    setCapturedPoses([false, false, false])
    setPoseOutcomes(null)
    setLiveError('')
  }, [])

  const allSaved =
    Array.isArray(poseOutcomes) &&
    poseOutcomes.length === POSE_SEQUENCE.length &&
    poseOutcomes.every((o) => o.status === 'saved')

  if (allSaved) {
    return (
      <div
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14, padding: '18px 0' }}
        role="status"
        aria-live="polite"
      >
        <CheckCircle size={40} style={{ color: 'var(--present-text)' }} aria-hidden="true" />
        <h3 style={{ margin: 0, fontFamily: 'Space Grotesk, sans-serif' }}>
          Guided enrollment complete
        </h3>
        <p style={{ margin: 0, color: 'var(--text-soft)' }}>
          All three poses were enrolled successfully for this student.
        </p>
        <a href={COVERAGE_URL} className="ghost-btn" style={{ textDecoration: 'none' }}>
          View enrollment coverage
        </a>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center' }}>
      {!cameraOn ? (
        <div className="empty-state" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
          <p style={{ fontWeight: 600 }}>Live guided mode</p>
          <small>Follow the on-screen poses — capture happens automatically.</small>
          <button type="button" className="solid-btn" onClick={handleEnableCamera} aria-label="Enable live camera">
            Enable Camera
          </button>
        </div>
      ) : (
        <>
          <div style={{ position: 'relative', width: '100%', maxWidth: 640 }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              aria-label="Live guided enrollment preview"
              style={{
                width: '100%',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border)',
                background: '#000',
                display: 'block',
              }}
            />
            <canvas
              ref={overlayCanvasRef}
              aria-hidden="true"
              style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
              }}
            />
          </div>

          {/* Reason chips from the latest preview evaluation */}
          <ul
            aria-label="Preview diagnostics"
            style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center', listStyle: 'none', margin: 0, padding: 0 }}
          >
            {previewChips.length === 0 && previewOk ? (
              <li key="ready" style={chipStyle('ok')}>
                Ready to capture
              </li>
            ) : (
              previewChips.map((reason) => (
                <li key={reason} style={chipStyle('warn')}>
                  {reason}
                </li>
              ))
            )}
          </ul>

          {/* Pose progress with per-pose checkmarks */}
          <ol
            aria-label="Guided pose progress"
            style={{ display: 'flex', gap: 10, flexWrap: 'wrap', justifyContent: 'center', listStyle: 'none', margin: 0, padding: 0 }}
          >
            {POSE_SEQUENCE.map((label, i) => {
              const captured = capturedPoses[i]
              const current = poseUi.poseIndex === i && !captured
              return (
                <li
                  key={label}
                  aria-current={current ? 'step' : undefined}
                  style={{
                    ...chipStyle(captured ? 'ok' : current ? 'warn' : 'idle'),
                    opacity: captured || current ? 1 : 0.6,
                  }}
                >
                  {captured && <Check size={13} aria-hidden="true" />}
                  <span>{label}</span>
                  {captured && <span aria-label="captured">✓</span>}
                </li>
              )
            })}
          </ol>

          {submittingLive && <p style={{ margin: 0, fontWeight: 600 }}>Submitting poses…</p>}

          {Array.isArray(poseOutcomes) && (
            <ul aria-label="Pose submission results" style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
              {poseOutcomes.map((outcome) => (
                <li key={outcome.pose} style={chipStyle(outcome.status === 'saved' ? 'ok' : 'warn')}>
                  {outcome.pose}: {outcome.status === 'saved' ? 'enrolled ✓' : outcome.message}
                </li>
              ))}
            </ul>
          )}

          <button type="button" className="ghost-btn" onClick={handleResetGuided} disabled={submittingLive}>
            <RefreshCcw size={14} aria-hidden="true" />
            <span>Redo guided capture</span>
          </button>

          {/* Hidden capture surfaces */}
          <canvas ref={workCanvasRef} style={{ display: 'none' }} />
          <canvas ref={fullCanvasRef} style={{ display: 'none' }} />
        </>
      )}

      {liveError && (
        <div className="alert-banner" role="alert">
          <AlertCircle size={16} aria-hidden="true" />
          <span>{liveError}</span>
        </div>
      )}
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function Enrollment() {
  const navigate = useNavigate()
  const { user } = useAuth()

  // step: 'pick' | 'capture' | 'submitting' | 'success'
  const [step, setStep] = useState('pick')
  const [selectedStudent, setSelectedStudent] = useState(null)
  const [capturedBlob, setCapturedBlob] = useState(null)
  const [submitError, setSubmitError] = useState('')
  const [successRecord, setSuccessRecord] = useState(null)
  const [qualityWarning, setQualityWarning] = useState(false)

  // New student modal state
  const [showNewStudentModal, setShowNewStudentModal] = useState(false)
  const [studentListKey, setStudentListKey] = useState(0)
  const [createdToast, setCreatedToast] = useState('')
  // Capture UX: 'manual' (snapshot button) or 'live' (guided auto-capture).
  const [captureMode, setCaptureMode] = useState('manual')

  const canCreateStudent = CREATE_STUDENT_ROLES.includes(user?.role)

  const handleStudentSelect = useCallback((student) => {
    setSelectedStudent(student)
    setSubmitError('')
    setStep('capture')
  }, [])

  const handleCapture = useCallback((blob) => {
    setCapturedBlob(blob)
    setSubmitError('')
  }, [])

  const handleRetake = useCallback(() => {
    setCapturedBlob(null)
    setSubmitError('')
    setQualityWarning(false)
  }, [])

  const handleSubmit = useCallback(async () => {
    if (!selectedStudent || !capturedBlob) return

    setStep('submitting')
    setSubmitError('')
    setQualityWarning(false)

    const formData = new FormData()
    formData.append('image_file', capturedBlob, 'enrollment.jpg')

    try {
      // ATT-034: never send a hand-rolled multipart Content-Type (it lacks
      // the boundary). Also strip the JSON default from the shared axios
      // instance — with a JSON content type axios serializes FormData into
      // `{"image_file":{}}` and silently drops the file.
      const response = await client.post(ENROLL_ENDPOINT(selectedStudent.id), formData, {
        headers: { 'Content-Type': undefined },
      })
      const record = response.data

      if (typeof record.quality_score === 'number' && record.quality_score < QUALITY_WARNING_THRESHOLD) {
        setQualityWarning(true)
      }

      setSuccessRecord({ ...record, studentName: selectedStudent.full_name })
      setStep('success')
    } catch (err) {
      if (err?.response?.status === 401) {
        navigate('/login')
        return
      }
      setSubmitError(resolveErrorMessage(err))
      setStep('capture')
    }
  }, [selectedStudent, capturedBlob, navigate])

  const handleEnrollAnother = useCallback(() => {
    setSelectedStudent(null)
    setCapturedBlob(null)
    setSuccessRecord(null)
    setSubmitError('')
    setQualityWarning(false)
    setStep('pick')
  }, [])

  const handleBackToPick = useCallback(() => {
    setSelectedStudent(null)
    setCapturedBlob(null)
    setSubmitError('')
    setStep('pick')
  }, [])

  const handleNewStudentCreated = useCallback((newStudent, fullName) => {
    setShowNewStudentModal(false)
    // Force StudentPicker to re-fetch by incrementing the key
    setStudentListKey((k) => k + 1)
    // Auto-select the newly created student
    setSelectedStudent(newStudent)
    setSubmitError('')
    setStep('capture')
    // Show toast for 6 seconds
    setCreatedToast(`Created ${fullName} · ready to enroll face`)
    setTimeout(() => setCreatedToast(''), 6000)
  }, [])

  return (
    <section className="surface-card">
      {/* ── Page header ──────────────────────────────────────────────── */}
      <header className="page-header">
        <div>
          <h2 className="page-title">Face Enrollment</h2>
          <p className="page-subtitle">
            Select a student, capture their face via webcam, and submit to create a biometric
            enrollment template.
          </p>
        </div>

        {/* Step indicator — 2 dots */}
        {step !== 'success' && step !== 'submitting' && (
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }} aria-hidden="true">
            {['pick', 'capture'].map((s, i) => (
              <div
                key={s}
                title={`Step ${i + 1}`}
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  background:
                    step === s
                      ? 'var(--brand)'
                      : step === 'capture' && s === 'pick'
                        ? 'var(--present-text)'
                        : 'rgba(15,23,42,0.18)',
                  transition: '0.2s ease',
                }}
              />
            ))}
          </div>
        )}
      </header>

      {/* ── Created-student toast ─────────────────────────────────────── */}
      {createdToast && (
        <div
          role="status"
          aria-live="polite"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '10px 14px',
            borderRadius: 'var(--radius-sm)',
            background: 'var(--present-bg)',
            border: '1px solid var(--present-border)',
            color: 'var(--present-text)',
            fontSize: '0.875rem',
            fontWeight: 500,
            marginBottom: 2,
          }}
        >
          <CheckCircle size={15} aria-hidden="true" style={{ flexShrink: 0 }} />
          <span>{createdToast}</span>
        </div>
      )}

      {/* ── Full-page loading overlay ─────────────────────────────────── */}
      {step === 'submitting' && <FullPageSpinner />}

      {/* ── Success state ─────────────────────────────────────────────── */}
      {step === 'success' && successRecord && (
        <SuccessCard
          record={successRecord}
          qualityWarning={qualityWarning}
          onEnrollAnother={handleEnrollAnother}
        />
      )}

      {/* ── Step 1: Student picker ─────────────────────────────────────── */}
      {step === 'pick' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Section header row: title + create button */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
              flexWrap: 'wrap',
            }}
          >
            <h3
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                fontFamily: 'Space Grotesk, sans-serif',
                fontSize: '1.05rem',
                fontWeight: 700,
                color: 'var(--text-main)',
                margin: 0,
              }}
            >
              <UserCheck size={18} style={{ color: 'var(--brand)' }} aria-hidden="true" />
              Step 1 — Select Student
            </h3>

            {canCreateStudent && (
              <button
                type="button"
                className="ghost-btn"
                onClick={() => setShowNewStudentModal(true)}
                style={{ flexShrink: 0 }}
              >
                <UserPlus size={14} aria-hidden="true" />
                <span>+ Create New Student</span>
              </button>
            )}
          </div>

          <StudentPicker
            key={studentListKey}
            selectedId={selectedStudent?.id ?? null}
            onSelect={handleStudentSelect}
            onNavigateToLogin={() => navigate('/login')}
          />
        </div>
      )}

      {/* ── New Student Modal ──────────────────────────────────────────── */}
      {showNewStudentModal && (
        <NewStudentModal
          onClose={() => setShowNewStudentModal(false)}
          onCreated={handleNewStudentCreated}
        />
      )}

      {/* ── Step 2: Webcam capture ─────────────────────────────────────── */}
      {step === 'capture' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Sub-header with back button */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              flexWrap: 'wrap',
            }}
          >
            <button type="button" className="ghost-btn" onClick={handleBackToPick}>
              Back
            </button>
            <h3
              style={{
                fontFamily: 'Space Grotesk, sans-serif',
                fontSize: '1.05rem',
                fontWeight: 700,
                color: 'var(--text-main)',
                margin: 0,
              }}
            >
              Step 2 — Capture Face
            </h3>
          </div>

          {/* Selected student chip */}
          {selectedStudent && (
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                padding: '7px 12px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--present-bg)',
                color: 'var(--present-text)',
                border: '1px solid rgba(18,102,70,0.22)',
                fontSize: '0.9rem',
                fontWeight: 600,
                alignSelf: 'flex-start',
              }}
              aria-label={`Selected student: ${selectedStudent.full_name}`}
            >
              <UserCheck size={15} aria-hidden="true" />
              <span>
                {selectedStudent.full_name}
                {selectedStudent.student_number ? ` · #${selectedStudent.student_number}` : ''}
              </span>
            </div>
          )}

          {/* Error banner */}
          {submitError && (
            <div className="alert-banner" role="alert">
              <AlertCircle size={16} aria-hidden="true" />
              <span>{submitError}</span>
            </div>
          )}

          {/* Mode switch: guided live capture vs manual snapshot */}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }} role="group" aria-label="Capture mode">
            <button
              type="button"
              className={captureMode === 'live' ? 'solid-btn' : 'ghost-btn'}
              onClick={() => setCaptureMode('live')}
              aria-pressed={captureMode === 'live'}
            >
              <span>Live Guided Mode</span>
            </button>
            <button
              type="button"
              className={captureMode === 'manual' ? 'solid-btn' : 'ghost-btn'}
              onClick={() => setCaptureMode('manual')}
              aria-pressed={captureMode === 'manual'}
            >
              <span>Manual Capture</span>
            </button>
          </div>

          {captureMode === 'live' ? (
            <LiveEnrollmentPanel studentId={selectedStudent?.id} />
          ) : (
            <WebcamCapture
              onCapture={handleCapture}
              capturedBlob={capturedBlob}
              onRetake={handleRetake}
              onSubmit={handleSubmit}
              isSubmitting={false}
            />
          )}
        </div>
      )}
    </section>
  )
}
