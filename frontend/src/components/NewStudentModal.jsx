import { useEffect, useRef, useState } from 'react'
import { AlertCircle, X } from 'lucide-react'
import client from '../api/client'

const CURRENT_YEAR = new Date().getFullYear()

// ── Password generator ────────────────────────────────────────────────────────
// Uses crypto.getRandomValues for cryptographic strength.
// 24 chars: upper + lower + digit + symbol, ensuring each class is present.
const UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
const LOWER = 'abcdefghijklmnopqrstuvwxyz'
const DIGIT = '0123456789'
const SYMBOL = '!@#$%^&*()-_=+[]{}|;:,.<>?'
const ALL = UPPER + LOWER + DIGIT + SYMBOL

function generatePassword(length = 24) {
  const array = new Uint32Array(length)
  crypto.getRandomValues(array)
  const chars = [
    UPPER[array[0] % UPPER.length],
    LOWER[array[1] % LOWER.length],
    DIGIT[array[2] % DIGIT.length],
    SYMBOL[array[3] % SYMBOL.length],
    ...Array.from({ length: length - 4 }, (_, i) => ALL[array[i + 4] % ALL.length]),
  ]
  // Shuffle using Fisher-Yates with fresh random values
  const shuffle = new Uint32Array(chars.length)
  crypto.getRandomValues(shuffle)
  for (let i = chars.length - 1; i > 0; i--) {
    const j = shuffle[i] % (i + 1)
    ;[chars[i], chars[j]] = [chars[j], chars[i]]
  }
  return chars.join('')
}

// ── Validation helpers ────────────────────────────────────────────────────────
const STUDENT_NUMBER_RE = /^[A-Za-z0-9][A-Za-z0-9_-]*$/

function validate(fields) {
  const errors = {}

  if (!fields.full_name.trim()) {
    errors.full_name = 'Full name is required.'
  } else if (fields.full_name.trim().length < 2 || fields.full_name.trim().length > 120) {
    errors.full_name = 'Full name must be 2–120 characters.'
  }

  if (!fields.email.trim()) {
    errors.email = 'Email is required.'
  } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(fields.email.trim())) {
    errors.email = 'Enter a valid email address.'
  }

  if (!fields.student_number.trim()) {
    errors.student_number = 'Student number is required.'
  } else if (
    fields.student_number.trim().length < 3 ||
    fields.student_number.trim().length > 32
  ) {
    errors.student_number = 'Student number must be 3–32 characters.'
  } else if (!STUDENT_NUMBER_RE.test(fields.student_number.trim())) {
    errors.student_number =
      'Only letters, digits, hyphens, and underscores; must start with a letter or digit.'
  }

  if (!fields.program.trim()) {
    errors.program = 'Program is required.'
  } else if (fields.program.trim().length < 2 || fields.program.trim().length > 120) {
    errors.program = 'Program must be 2–120 characters.'
  }

  const enrollYear = parseInt(fields.enrollment_year, 10)
  if (!fields.enrollment_year || isNaN(enrollYear) || enrollYear < 2000 || enrollYear > 2100) {
    errors.enrollment_year = 'Enter a valid enrollment year (2000–2100).'
  }

  if (fields.graduation_year) {
    const gradYear = parseInt(fields.graduation_year, 10)
    if (isNaN(gradYear) || gradYear < 2000 || gradYear > 2100) {
      errors.graduation_year = 'Enter a valid graduation year (2000–2100).'
    } else if (!isNaN(enrollYear) && gradYear < enrollYear) {
      errors.graduation_year = 'Graduation year must be ≥ enrollment year.'
    }
  }

  return errors
}

// ── Modal ─────────────────────────────────────────────────────────────────────

export default function NewStudentModal({ onClose, onCreated }) {
  const [fields, setFields] = useState({
    full_name: '',
    email: '',
    student_number: '',
    program: '',
    enrollment_year: String(CURRENT_YEAR),
    graduation_year: '',
    date_of_birth: '',
  })
  const [touched, setTouched] = useState({})
  const [submitError, setSubmitError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const firstInputRef = useRef(null)
  const backdropRef = useRef(null)

  // Focus trap on mount
  useEffect(() => {
    firstInputRef.current?.focus()
  }, [])

  // Close on Escape
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  const handleBackdropClick = (e) => {
    if (e.target === backdropRef.current) onClose()
  }

  const setField = (name, value) => {
    setFields((prev) => ({ ...prev, [name]: value }))
  }

  const markTouched = (name) => {
    setTouched((prev) => ({ ...prev, [name]: true }))
  }

  const errors = validate(fields)
  const showError = (name) => (touched[name] ? errors[name] : undefined)

  const handleSubmit = async (e) => {
      e.preventDefault()

      // Mark all fields touched to show all validation errors
      setTouched({
        full_name: true,
        email: true,
        student_number: true,
        program: true,
        enrollment_year: true,
        graduation_year: true,
        date_of_birth: true,
      })

      if (Object.keys(errors).length > 0) return

      setIsSubmitting(true)
      setSubmitError('')

      const password = generatePassword(24)

      // Step 1: Create user
      let userId
      try {
        const userRes = await client.post('/api/v1/users', {
          email: fields.email.trim(),
          full_name: fields.full_name.trim(),
          password,
          role: 'auditor',
        })
        userId = userRes.data.id
      } catch (err) {
        const detail =
          err?.response?.data?.detail ||
          err?.message ||
          'User creation failed. Check the email and try again.'
        setSubmitError(detail)
        setIsSubmitting(false)
        return
      }

      // Step 2: Create student
      const studentBody = {
        user_id: userId,
        student_number: fields.student_number.trim(),
        program: fields.program.trim(),
        enrollment_year: parseInt(fields.enrollment_year, 10),
        is_active: true,
      }
      if (fields.graduation_year) {
        studentBody.graduation_year = parseInt(fields.graduation_year, 10)
      }
      if (fields.date_of_birth) {
        studentBody.date_of_birth = fields.date_of_birth
      }

      let newStudent
      try {
        const studentRes = await client.post('/api/v1/students', studentBody)
        newStudent = { ...studentRes.data, full_name: fields.full_name.trim() }
      } catch (err) {
        const detail =
          err?.response?.data?.detail ||
          err?.message ||
          'Student record creation failed.'
        setSubmitError(
          `${detail} — User created but student record failed. Contact admin to clean up: user id ${userId}.`,
        )
        setIsSubmitting(false)
        return
      }

      setIsSubmitting(false)
      onCreated(newStudent, fields.full_name.trim())
  }

  return (
    <div
      ref={backdropRef}
      onClick={handleBackdropClick}
      role="dialog"
      aria-modal="true"
      aria-labelledby="new-student-modal-title"
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(15, 23, 42, 0.6)',
        display: 'grid',
        placeItems: 'center',
        zIndex: 50,
        padding: '16px',
      }}
    >
      <div
        className="surface-card"
        style={{
          width: 'min(560px, 100%)',
          maxHeight: 'calc(100vh - 32px)',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 20,
          padding: 24,
        }}
        role="document"
      >
        {/* Modal header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
          }}
        >
          <h2
            id="new-student-modal-title"
            style={{
              fontFamily: 'Space Grotesk, Inter, sans-serif',
              fontSize: '1.1rem',
              fontWeight: 700,
              color: 'var(--text-main)',
              margin: 0,
            }}
          >
            Create New Student
          </h2>
          <button
            type="button"
            className="icon-btn"
            onClick={onClose}
            aria-label="Close modal"
          >
            <X size={16} aria-hidden="true" />
          </button>
        </div>

        {/* Submit error */}
        {submitError && (
          <div className="alert-banner" role="alert" style={{ marginBottom: 0 }}>
            <AlertCircle size={16} aria-hidden="true" style={{ flexShrink: 0 }} />
            <span>{submitError}</span>
          </div>
        )}

        <form
          onSubmit={handleSubmit}
          noValidate
          style={{ display: 'flex', flexDirection: 'column', gap: 14 }}
        >
          {/* Full Name */}
          <FieldBlock label="Full Name" required error={showError('full_name')}>
            <input
              ref={firstInputRef}
              type="text"
              className="input-field"
              value={fields.full_name}
              onChange={(e) => setField('full_name', e.target.value)}
              onBlur={() => markTouched('full_name')}
              placeholder="Jane Doe"
              maxLength={120}
              aria-required="true"
              aria-describedby={showError('full_name') ? 'err-full_name' : undefined}
            />
          </FieldBlock>

          {/* Email */}
          <FieldBlock
            label="Email"
            required
            error={showError('email')}
            hint="Used only for record-keeping; auto-generated login password."
          >
            <input
              type="email"
              className="input-field"
              value={fields.email}
              onChange={(e) => setField('email', e.target.value)}
              onBlur={() => markTouched('email')}
              placeholder="jane.doe@university.edu"
              aria-required="true"
              aria-describedby={showError('email') ? 'err-email' : 'hint-email'}
            />
          </FieldBlock>

          {/* Student Number */}
          <FieldBlock label="Student Number" required error={showError('student_number')}>
            <input
              type="text"
              className="input-field"
              value={fields.student_number}
              onChange={(e) => setField('student_number', e.target.value)}
              onBlur={() => markTouched('student_number')}
              placeholder="S2026099"
              maxLength={32}
              aria-required="true"
              aria-describedby={showError('student_number') ? 'err-student_number' : undefined}
            />
          </FieldBlock>

          {/* Program */}
          <FieldBlock label="Program" required error={showError('program')}>
            <input
              type="text"
              className="input-field"
              value={fields.program}
              onChange={(e) => setField('program', e.target.value)}
              onBlur={() => markTouched('program')}
              placeholder="Computer Science"
              maxLength={120}
              aria-required="true"
              aria-describedby={showError('program') ? 'err-program' : undefined}
            />
          </FieldBlock>

          {/* Enrollment Year + Graduation Year side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <FieldBlock label="Enrollment Year" required error={showError('enrollment_year')}>
              <input
                type="number"
                className="input-field"
                value={fields.enrollment_year}
                onChange={(e) => setField('enrollment_year', e.target.value)}
                onBlur={() => markTouched('enrollment_year')}
                min={2000}
                max={2100}
                placeholder={String(CURRENT_YEAR)}
                aria-required="true"
                aria-describedby={showError('enrollment_year') ? 'err-enrollment_year' : undefined}
              />
            </FieldBlock>

            <FieldBlock label="Graduation Year" error={showError('graduation_year')}>
              <input
                type="number"
                className="input-field"
                value={fields.graduation_year}
                onChange={(e) => setField('graduation_year', e.target.value)}
                onBlur={() => markTouched('graduation_year')}
                min={2000}
                max={2100}
                placeholder="Optional"
                aria-describedby={showError('graduation_year') ? 'err-graduation_year' : undefined}
              />
            </FieldBlock>
          </div>

          {/* Date of Birth */}
          <FieldBlock label="Date of Birth" error={showError('date_of_birth')}>
            <input
              type="date"
              className="input-field"
              value={fields.date_of_birth}
              onChange={(e) => setField('date_of_birth', e.target.value)}
              onBlur={() => markTouched('date_of_birth')}
              aria-describedby={showError('date_of_birth') ? 'err-date_of_birth' : undefined}
            />
          </FieldBlock>

          {/* Actions */}
          <div
            style={{
              display: 'flex',
              gap: 10,
              justifyContent: 'flex-end',
              paddingTop: 4,
              borderTop: '1px solid var(--border)',
              marginTop: 4,
            }}
          >
            <button
              type="button"
              className="ghost-btn"
              onClick={onClose}
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="solid-btn"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Creating...' : 'Create & select'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

// ── Field block helper ────────────────────────────────────────────────────────
function FieldBlock({ label, required, error, hint, children }) {
  const fieldId = label.toLowerCase().replace(/\s+/g, '_')
  return (
    <label
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 5,
        fontSize: '0.82rem',
        fontWeight: 600,
        color: 'var(--text-soft)',
      }}
    >
      <span>
        {label}
        {required && (
          <span aria-hidden="true" style={{ color: 'var(--absent-text)', marginLeft: 3 }}>
            *
          </span>
        )}
      </span>
      {children}
      {hint && !error && (
        <span
          id={`hint-${fieldId}`}
          style={{ fontSize: '0.77rem', color: 'var(--text-subtle)', fontWeight: 400 }}
        >
          {hint}
        </span>
      )}
      {error && (
        <span
          id={`err-${fieldId}`}
          role="alert"
          style={{ fontSize: '0.77rem', color: 'var(--absent-text)', fontWeight: 500 }}
        >
          {error}
        </span>
      )}
    </label>
  )
}
