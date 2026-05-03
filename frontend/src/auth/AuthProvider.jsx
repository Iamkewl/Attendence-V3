import { useCallback, useEffect, useMemo, useState } from 'react'
import client from '../api/client'
import { AuthContext } from './AuthContext'

const getApiErrorMessage = (error, fallbackMessage) =>
  error?.response?.data?.detail || fallbackMessage

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [authError, setAuthError] = useState('')

  const clearAuthError = useCallback(() => {
    setAuthError('')
  }, [])

  const refreshSession = useCallback(async ({ silent = false } = {}) => {
    try {
      const response = await client.post('/api/v1/auth/refresh')
      setUser(response.data.user)
      setAuthError('')
      return response.data.user
    } catch (error) {
      setUser(null)
      if (!silent) {
        setAuthError(getApiErrorMessage(error, 'Unable to refresh session.'))
      }
      throw error
    }
  }, [])

  const login = useCallback(async ({ email, password }) => {
    try {
      const response = await client.post('/api/v1/auth/login', { email, password })
      setUser(response.data.user)
      setAuthError('')
      return response.data.user
    } catch (error) {
      const message = getApiErrorMessage(error, 'Login failed.')
      setAuthError(message)
      throw new Error(message, { cause: error })
    }
  }, [])

  const logout = useCallback(async () => {
    try {
      await client.post('/api/v1/auth/logout', null, { skipAuthRefresh: true })
    } finally {
      setUser(null)
      setAuthError('')
    }
  }, [])

  useEffect(() => {
    let active = true

    const bootstrapSession = async () => {
      try {
        await refreshSession({ silent: true })
      } catch {
        // Unauthenticated startup is expected until user signs in.
      } finally {
        if (active) {
          setIsLoading(false)
        }
      }
    }

    bootstrapSession()

    return () => {
      active = false
    }
  }, [refreshSession])

  const value = useMemo(
    () => ({
      user,
      isAuthenticated: Boolean(user),
      isLoading,
      authError,
      clearAuthError,
      login,
      logout,
      refreshSession,
    }),
    [user, isLoading, authError, clearAuthError, login, logout, refreshSession],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
