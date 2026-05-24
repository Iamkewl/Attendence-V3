import { useState } from 'react'
import { KeyRound, ShieldCheck } from 'lucide-react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export default function LoginPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const { login, isAuthenticated, isLoading, authError, clearAuthError } = useAuth()

  const [formState, setFormState] = useState({
    email: '',
    password: '',
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [formError, setFormError] = useState('')

  const requestedPath =
    typeof location.state?.from === 'string' && location.state.from.length > 0
      ? location.state.from
      : '/dashboard'

  const nextPath = requestedPath === '/login' ? '/dashboard' : requestedPath

  if (isLoading) {
    return (
      <div className="auth-screen">
        <div className="auth-card compact">
          <div className="auth-logo-mark" aria-hidden="true">A3</div>
          <h1>Attendance V3</h1>
          <p>Checking session…</p>
        </div>
      </div>
    )
  }

  if (isAuthenticated) {
    return <Navigate to={nextPath} replace />
  }

  const onInputChange = (event) => {
    const { name, value } = event.target
    setFormState((previous) => ({ ...previous, [name]: value }))
    // Clear errors when user starts typing
    if (formError) setFormError('')
    if (authError) clearAuthError()
  }

  const onSubmit = async (event) => {
    event.preventDefault()
    setFormError('')
    clearAuthError()
    setIsSubmitting(true)

    try {
      await login(formState)
      navigate(nextPath, { replace: true })
    } catch (error) {
      setFormError(error.message || 'Unable to sign in.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const displayError = formError || authError

  return (
    <div className="auth-screen">
      <form className="auth-card" onSubmit={onSubmit} noValidate aria-label="Sign in form">
        {/* Brand header */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div className="auth-logo-mark" aria-hidden="true">A3</div>
          <div>
            <h1>Sign in to Attendance V3</h1>
            <p style={{ marginTop: '6px' }}>
              Use your instructor credentials to access the dashboard.
            </p>
          </div>
        </div>

        <div className="auth-divider" />

        {/* Form fields */}
        <div className="auth-form">
          <label htmlFor="login-email">
            Email address
            <input
              id="login-email"
              className="input-field"
              type="email"
              name="email"
              autoComplete="username"
              value={formState.email}
              onChange={onInputChange}
              placeholder="instructor@example.edu"
              required
              aria-required="true"
            />
          </label>

          <label htmlFor="login-password">
            Password
            <input
              id="login-password"
              className="input-field"
              type="password"
              name="password"
              autoComplete="current-password"
              value={formState.password}
              onChange={onInputChange}
              placeholder="••••••••"
              required
              aria-required="true"
              minLength={8}
            />
          </label>
        </div>

        {/* Error banner */}
        {displayError ? (
          <div className="alert-banner" role="alert" aria-live="assertive">
            <ShieldCheck size={15} aria-hidden="true" />
            <span>{displayError}</span>
          </div>
        ) : null}

        {/* Submit */}
        <button
          className="solid-btn"
          type="submit"
          disabled={isSubmitting}
          style={{ width: '100%', justifyContent: 'center', padding: '10px 14px' }}
          aria-busy={isSubmitting}
        >
          <KeyRound size={15} aria-hidden="true" />
          <span>{isSubmitting ? 'Signing in…' : 'Sign In'}</span>
        </button>

        <p style={{ fontSize: '0.78rem', color: 'var(--text-subtle)', textAlign: 'center', marginTop: '-6px' }}>
          Session is secured via HTTP-only cookies.
        </p>
      </form>
    </div>
  )
}
