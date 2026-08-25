import { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, CheckCircle, RefreshCcw, UserCheck, UserPlus } from 'lucide-react'
import client from '../api/client'
import { useAuth } from '../auth/AuthContext'
import WebcamCapture from './WebcamCapture'
import StudentPicker from './StudentPicker'
import NewStudentModal from '../components/NewStudentModal'

const CREATE_STUDENT_ROLES = ['admin', 'instructor']

const ENROLL_ENDPOINT = (studentId) => `/api/v1/students/${studentId}/enroll`
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

          <WebcamCapture
            onCapture={handleCapture}
            capturedBlob={capturedBlob}
            onRetake={handleRetake}
            onSubmit={handleSubmit}
            isSubmitting={false}
          />
        </div>
      )}
    </section>
  )
}
