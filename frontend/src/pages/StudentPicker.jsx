import { useCallback, useEffect, useState } from 'react'
import { Search, User } from 'lucide-react'
import client from '../api/client'

const STUDENTS_ENDPOINT = '/api/v1/students'

/**
 * StudentPicker
 *
 * Props:
 *   selectedId: string | null
 *   onSelect(student) — called when a row is chosen
 *   onNavigateToLogin() — called on 401
 */
export default function StudentPicker({ selectedId, onSelect, onNavigateToLogin }) {
  const [students, setStudents] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [query, setQuery] = useState('')
  const [useFallback, setUseFallback] = useState(false)
  const [manualId, setManualId] = useState('')

  const loadStudents = useCallback(async () => {
    setIsLoading(true)
    setLoadError('')
    try {
      const response = await client.get(STUDENTS_ENDPOINT, { params: { limit: 200 } })
      setStudents(Array.isArray(response.data) ? response.data : [])
    } catch (err) {
      if (err?.response?.status === 401) {
        onNavigateToLogin()
        return
      }
      setLoadError(err?.response?.data?.detail || err.message || 'Unable to load students.')
    } finally {
      setIsLoading(false)
    }
  }, [onNavigateToLogin])

  useEffect(() => {
    if (!useFallback) {
      // loadStudents is a stable useCallback; setState calls happen inside an
      // async callback, not synchronously in the effect body.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      loadStudents()
    }
  }, [useFallback, loadStudents])

  const normalizedQuery = query.toLowerCase()
  const filtered = students.filter((s) => {
    const name = String(s.full_name || '').toLowerCase()
    const number = String(s.student_number || '').toLowerCase()
    const id = String(s.id || '').toLowerCase()
    return (
      name.includes(normalizedQuery) ||
      number.includes(normalizedQuery) ||
      id.includes(normalizedQuery)
    )
  })

  const handleManualSubmit = (e) => {
    e.preventDefault()
    const trimmed = manualId.trim()
    if (!trimmed) return
    onSelect({ id: trimmed, full_name: trimmed, student_number: '' })
  }

  if (useFallback) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: '0.86rem', color: 'var(--text-soft)', fontWeight: 600 }}>
          Student UUID
          <form onSubmit={handleManualSubmit} style={{ display: 'flex', gap: 8 }}>
            <input
              className="input-field"
              type="text"
              placeholder="Paste student UUID (e.g. 3fa85f64-...)"
              value={manualId}
              onChange={(e) => setManualId(e.target.value)}
              aria-label="Student UUID"
              required
            />
            <button type="submit" className="solid-btn">
              <User size={14} aria-hidden="true" />
              <span>Select</span>
            </button>
          </form>
        </label>
        <button
          type="button"
          className="ghost-btn"
          onClick={() => setUseFallback(false)}
          style={{ alignSelf: 'flex-start' }}
        >
          Back to student list
        </button>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {/* Search bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          border: '1px solid rgba(15,23,42,0.2)',
          borderRadius: 'var(--radius-sm)',
          padding: '8px 11px',
          background: '#fff',
        }}
      >
        <Search size={15} style={{ color: 'var(--text-subtle)', flexShrink: 0 }} aria-hidden="true" />
        <input
          type="search"
          placeholder="Search by name or student number..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search students"
          style={{
            border: 'none',
            outline: 'none',
            flex: 1,
            background: 'transparent',
            fontSize: '0.94rem',
            color: 'var(--text-main)',
          }}
        />
      </div>

      {isLoading ? (
        <p className="meta-note" style={{ padding: '8px 0' }}>Loading students...</p>
      ) : loadError ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div className="alert-banner" role="alert">
            <span>{loadError}</span>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="button" className="ghost-btn" onClick={loadStudents}>
              Retry
            </button>
            <button type="button" className="ghost-btn" onClick={() => setUseFallback(true)}>
              Enter ID manually
            </button>
          </div>
        </div>
      ) : filtered.length === 0 ? (
        <div className="empty-state compact">
          <p>{query ? 'No students match your search.' : 'No students found.'}</p>
          <button
            type="button"
            className="ghost-btn"
            style={{ marginTop: 8 }}
            onClick={() => setUseFallback(true)}
          >
            Enter ID manually
          </button>
        </div>
      ) : (
        <ul
          role="listbox"
          aria-label="Student list"
          style={{
            listStyle: 'none',
            margin: 0,
            padding: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
            maxHeight: 320,
            overflowY: 'auto',
          }}
        >
          {filtered.map((student) => {
            const isSelected = student.id === selectedId
            return (
              <li
                key={student.id}
                role="option"
                aria-selected={isSelected}
                tabIndex={0}
                onClick={() => onSelect(student)}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onSelect(student)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-sm)',
                  border: `1px solid ${isSelected ? 'rgba(231,134,53,0.5)' : 'var(--border)'}`,
                  background: isSelected ? 'rgba(231,134,53,0.08)' : 'var(--surface-strong)',
                  cursor: 'pointer',
                  transition: '0.14s ease',
                  outline: 'none',
                }}
              >
                <User
                  size={15}
                  style={{ flexShrink: 0, color: isSelected ? 'var(--brand)' : 'var(--text-subtle)' }}
                  aria-hidden="true"
                />
                <div>
                  <p style={{ fontWeight: 600, fontSize: '0.92rem', color: 'var(--text-main)', margin: 0 }}>
                    {student.full_name || 'Unnamed'}
                  </p>
                  {student.student_number ? (
                    <small style={{ color: 'var(--text-subtle)' }}>#{student.student_number}</small>
                  ) : null}
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
