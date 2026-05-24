import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, Calendar, ClipboardList, Filter, RefreshCcw } from 'lucide-react'
import client from '../api/client'

const ATTENDANCE_SESSIONS_ENDPOINT = '/api/v1/attendance/sessions'

const STATUS_PRESENT = 'present'
const STATUS_ABSENT = 'absent'
const STATUS_PENDING = 'pending'

const STATUS_PILL_CLASS = {
  [STATUS_PRESENT]: 'status-pill present',
  [STATUS_ABSENT]: 'status-pill absent',
  [STATUS_PENDING]: 'status-pill pending',
}

// Sort order: present → absent → pending → unknown
const STATUS_SORT_ORDER = {
  [STATUS_PRESENT]: 0,
  [STATUS_ABSENT]: 1,
  [STATUS_PENDING]: 2,
}

const mapStatusLabel = (status = '') => {
  const normalized = String(status).toLowerCase()
  if (normalized === STATUS_PRESENT) return 'Present'
  if (normalized === STATUS_ABSENT) return 'Absent'
  if (normalized === STATUS_PENDING) return 'Pending'
  return status || 'Unknown'
}

const deriveStatus = (record) => {
  const normalizedStatus = String(record?.status ?? '').toLowerCase()
  if (normalizedStatus) return normalizedStatus

  const sightingCount = Number(record?.sightings_count ?? 0)
  // Use the new schema field; fall back to legacy name if somehow still present
  const threshold = Number(record?.required_sightings_threshold ?? record?.required_sightings ?? 1)

  if (Number.isFinite(sightingCount) && Number.isFinite(threshold)) {
    return sightingCount >= threshold ? STATUS_PRESENT : STATUS_ABSENT
  }

  return 'unknown'
}

const getRecordRowKey = (record, index) => {
  if (record?.id) return record.id
  return [
    record?.student_id ?? 'unknown-student',
    record?.course_id ?? 'unknown-course',
    record?.session_date ?? 'unknown-session',
    index,
  ].join(':')
}

function formatTimestamp(value) {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return 'Not evaluated'
  return parsed.toLocaleString()
}

export default function ClassRoster() {
  const navigate = useNavigate()
  const [records, setRecords] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [lastEvalAt, setLastEvalAt] = useState(null)
  const [sessionMeta, setSessionMeta] = useState(null)
  const [filters, setFilters] = useState({
    courseId: import.meta.env.VITE_DEMO_COURSE_ID || '',
    sessionDate: new Date().toISOString().split('T')[0],
    status: 'all',
  })

  const queryParams = useMemo(() => {
    const params = {}
    const trimmedCourseId = filters.courseId.trim()
    if (trimmedCourseId) params.course_id = trimmedCourseId
    if (filters.sessionDate) params.session_date = filters.sessionDate
    if (filters.status !== 'all') params.status = filters.status
    return params
  }, [filters])

  const loadRoster = useCallback(async () => {
    setIsLoading(true)
    setLoadError('')

    try {
      const response = await client.get(ATTENDANCE_SESSIONS_ENDPOINT, { params: queryParams })
      // New API: { course_id, session_date, required_sightings_threshold, records: [...] }
      const payload = response.data
      const rawRecords = Array.isArray(payload?.records) ? payload.records : []

      // Capture session-level metadata for the summary banner
      setSessionMeta({
        courseId: payload?.course_id || queryParams.course_id || null,
        sessionDate: payload?.session_date || queryParams.session_date || null,
        threshold: payload?.required_sightings_threshold ?? null,
      })

      // Derive statuses and sort: present first, then absent, then pending/unknown
      const normalizedRecords = rawRecords
        .map((record) => ({
          ...record,
          status: deriveStatus(record),
        }))
        .sort((a, b) => {
          // Primary: status order (present → absent → pending)
          const aOrder = STATUS_SORT_ORDER[a.status] ?? 3
          const bOrder = STATUS_SORT_ORDER[b.status] ?? 3
          if (aOrder !== bOrder) return aOrder - bOrder
          // Secondary: most recently evaluated first
          const aTime = new Date(a.evaluated_at || a.updated_at || 0).getTime()
          const bTime = new Date(b.evaluated_at || b.updated_at || 0).getTime()
          return bTime - aTime
        })

      // Find the most recent evaluation timestamp across all records
      let newestEval = null
      for (const rec of normalizedRecords) {
        const t = rec.evaluated_at || rec.updated_at
        if (t) {
          const parsed = new Date(t)
          if (!newestEval || parsed > newestEval) newestEval = parsed
        }
      }
      setLastEvalAt(newestEval)
      setRecords(normalizedRecords)
    } catch (error) {
      const statusCode = error?.response?.status

      if (statusCode === 401) {
        navigate('/login')
        return
      }

      if (statusCode === 404) {
        setLoadError('Course not found.')
        setRecords([])
        setLastEvalAt(null)
        return
      }

      const detail = error?.response?.data?.detail
      setLoadError(detail || error.message || 'Unable to load class roster.')
      setRecords([])
      setLastEvalAt(null)
    } finally {
      setIsLoading(false)
    }
  }, [queryParams, navigate])

  useEffect(() => {
    const timerId = window.setTimeout(() => {
      loadRoster()
    }, 0)
    return () => window.clearTimeout(timerId)
  }, [loadRoster])

  const onFilterChange = (event) => {
    const { name, value } = event.target
    setFilters((previous) => ({ ...previous, [name]: value }))
  }

  const onApplyFilters = (event) => {
    event.preventDefault()
    loadRoster()
  }

  // Summary counts
  const counts = useMemo(() => {
    const result = { present: 0, absent: 0, pending: 0 }
    for (const rec of records) {
      if (rec.status === STATUS_PRESENT) result.present++
      else if (rec.status === STATUS_ABSENT) result.absent++
      else result.pending++
    }
    return result
  }, [records])

  return (
    <section className="surface-card" aria-labelledby="roster-title">
      <header className="page-header">
        <div>
          <h2 className="page-title" id="roster-title">Class Session Roster</h2>
          <p className="page-subtitle">
            Final attendance outcomes derived from aggregated sighting counts per student and class
            session date.
          </p>
        </div>

        <button
          type="button"
          className="ghost-btn"
          onClick={loadRoster}
          disabled={isLoading}
          aria-label="Refresh roster"
        >
          <RefreshCcw size={13} aria-hidden="true" />
          <span>{isLoading ? 'Refreshing...' : 'Refresh'}</span>
        </button>
      </header>

      {/* Filter form */}
      <form className="filters-grid" onSubmit={onApplyFilters} aria-label="Roster filters">
        <label htmlFor="roster-course-id">
          Course ID
          <input
            id="roster-course-id"
            className="input-field"
            name="courseId"
            value={filters.courseId}
            onChange={onFilterChange}
            placeholder="e.g. CS101 or UUID"
            autoComplete="off"
          />
        </label>

        <label htmlFor="roster-date">
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Calendar size={12} aria-hidden="true" />
            Session Date
          </span>
          <input
            id="roster-date"
            className="input-field"
            type="date"
            name="sessionDate"
            value={filters.sessionDate}
            onChange={onFilterChange}
          />
        </label>

        <label htmlFor="roster-status">
          Status
          <select
            id="roster-status"
            className="input-field"
            name="status"
            value={filters.status}
            onChange={onFilterChange}
          >
            <option value="all">All statuses</option>
            <option value="PRESENT">Present</option>
            <option value="ABSENT">Absent</option>
            <option value="PENDING">Pending</option>
          </select>
        </label>

        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button type="submit" className="solid-btn" disabled={isLoading} style={{ width: '100%', justifyContent: 'center' }}>
            <Filter size={13} aria-hidden="true" />
            <span>Apply</span>
          </button>
        </div>
      </form>

      {/* Last-evaluation summary banner */}
      {records.length > 0 && (
        <div className="eval-banner" role="status" aria-label="Session evaluation summary">
          <ClipboardList size={15} aria-hidden="true" />
          <span>
            {sessionMeta?.courseId && (
              <>Course: <strong>{sessionMeta.courseId}</strong> &middot; </>
            )}
            {sessionMeta?.sessionDate && (
              <>Date: <strong>{sessionMeta.sessionDate}</strong> &middot; </>
            )}
            <strong>{counts.present}</strong> present &middot; <strong>{counts.absent}</strong> absent &middot; <strong>{counts.pending}</strong> pending
            {lastEvalAt && (
              <> &middot; Last evaluated: <strong>{lastEvalAt.toLocaleString()}</strong></>
            )}
          </span>
        </div>
      )}

      {loadError ? (
        <div className="alert-banner" role="alert">
          <AlertCircle size={15} aria-hidden="true" />
          <span>{loadError}</span>
        </div>
      ) : null}

      <div className="table-wrap" role="region" aria-label="Attendance records">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Student</th>
              <th scope="col">Sightings / Required</th>
              <th scope="col">Status</th>
              <th scope="col">Evaluated At</th>
            </tr>
          </thead>

          <tbody>
            {isLoading && records.length === 0 ? (
              <tr>
                <td colSpan={4}>
                  <div className="empty-state compact">
                    <p>Loading records…</p>
                  </div>
                </td>
              </tr>
            ) : records.length === 0 ? (
              <tr>
                <td colSpan={4}>
                  <div className="empty-state compact">
                    <ClipboardList size={24} style={{ color: 'var(--text-subtle)', marginBottom: '6px' }} aria-hidden="true" />
                    <p>No attendance records found.</p>
                    <small>
                      {filters.courseId
                        ? 'Try a different course or date, or wait for evaluation to run.'
                        : 'Select a course and date, then click Apply.'}
                    </small>
                  </div>
                </td>
              </tr>
            ) : (
              records.map((record, index) => (
                <tr key={getRecordRowKey(record, index)}>
                  <td style={{ fontWeight: 500 }}>
                    {record.student_full_name || record.student_id}
                  </td>
                  <td>
                    <span style={{ fontVariantNumeric: 'tabular-nums' }}>
                      {record.sightings_count ?? 0}
                      {' / '}
                      {record.required_sightings_threshold ?? 0}
                    </span>
                  </td>
                  <td>
                    <span
                      className={
                        STATUS_PILL_CLASS[String(record.status).toLowerCase()] ||
                        'status-pill pending'
                      }
                    >
                      {mapStatusLabel(record.status)}
                    </span>
                  </td>
                  <td style={{ color: 'var(--text-soft)', fontSize: '0.84rem' }}>
                    {record.evaluated_at
                      ? formatTimestamp(record.evaluated_at)
                      : 'Not evaluated'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {records.length > 0 && (
        <p className="meta-note" style={{ marginTop: '10px' }}>
          {records.length} record{records.length !== 1 ? 's' : ''} — sorted by status (present first)
        </p>
      )}
    </section>
  )
}
