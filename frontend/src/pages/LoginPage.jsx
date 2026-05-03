import { useState } from 'react'
import { KeyRound } from 'lucide-react'
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
      : '/heartbeat-feed'

  const nextPath = requestedPath === '/login' ? '/heartbeat-feed' : requestedPath

  if (isLoading) {
    return (
      <div className="auth-screen">
        <div className="auth-card compact">
          <h1>Attendance V2</h1>
          <p>Checking session...</p>
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

  return (
    <div className="auth-screen">
      <form className="auth-card" onSubmit={onSubmit}>
        <h1>Sign In</h1>
        <p>Use your secure cookie-backed instructor account.</p>

        <div className="auth-form">
          <label>
            Email
            <input
              className="input-field"
              type="email"
              name="email"
              autoComplete="username"
              value={formState.email}
              onChange={onInputChange}
              required
            />
          </label>

          <label>
            Password
            <input
              className="input-field"
              type="password"
              name="password"
              autoComplete="current-password"
              value={formState.password}
              onChange={onInputChange}
              required
              minLength={8}
            />
          </label>
        </div>

        {formError || authError ? (
          <div className="alert-banner" role="alert">
            <span>{formError || authError}</span>
          </div>
        ) : null}

        <button className="solid-btn" type="submit" disabled={isSubmitting}>
          <KeyRound size={15} />
          <span>{isSubmitting ? 'Signing In...' : 'Sign In'}</span>
        </button>
      </form>
    </div>
  )
}