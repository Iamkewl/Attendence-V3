import { useCallback, useEffect, useMemo, useState } from 'react'
import { AlertCircle, Filter, RefreshCcw } from 'lucide-react'
import client from '../api/client'

const ATTENDANCE_SESSIONS_ENDPOINT = '/api/v1/attendance/sessions'

const mapStatusLabel = (status = '') => {
  const normalized = String(status).toLowerCase()

  if (normalized === 'present') return 'Present'
  if (normalized === 'absent') return 'Absent'
  if (normalized === 'late') return 'Late'
  if (normalized === 'excused') return 'Excused'

  return status || 'Unknown'
}

const deriveStatus = (record) => {
  const normalizedStatus = String(record?.status ?? '').toLowerCase()
  if (normalizedStatus) {
    return normalizedStatus
  }

  const sightingCount = Number(record?.sighting_count ?? 0)
  const threshold = Number(record?.required_sightings ?? 1)

  if (Number.isFinite(sightingCount) && Number.isFinite(threshold)) {
    return sightingCount >= threshold ? 'present' : 'absent'
  }

  return 'unknown'
}

const getRecordRowKey = (record, index) => {
  if (record?.id) {
    return record.id
  }

  return [
    record?.student_id ?? 'unknown-student',
    record?.course_id ?? 'unknown-course',
    record?.session_date ?? 'unknown-session',
    index,
  ].join(':')
}

export default function ClassRoster() {
  const [records, setRecords] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [filters, setFilters] = useState({
    courseId: '',
    sessionDate: '',
    status: 'all',
  })

  const queryParams = useMemo(() => {
    const params = {}
    const trimmedCourseId = filters.courseId.trim()

    if (trimmedCourseId) {
      params.course_id = trimmedCourseId
    }

    if (filters.sessionDate) {
      params.session_date = filters.sessionDate
    }

    if (filters.status !== 'all') {
      params.status = filters.status
    }

    return params
  }, [filters])

  const loadRoster = useCallback(async () => {
    setIsLoading(true)
    setLoadError('')

    try {
      const response = await client.get(ATTENDANCE_SESSIONS_ENDPOINT, { params: queryParams })
      const payload = Array.isArray(response.data) ? response.data : []

      const normalizedRecords = payload
        .map((record) => ({
          ...record,
          status: deriveStatus(record),
        }))
        .sort((a, b) => {
          const leftTime = new Date(b.evaluated_at || b.updated_at || 0).getTime()
          const rightTime = new Date(a.evaluated_at || a.updated_at || 0).getTime()
          return leftTime - rightTime
        })

      setRecords(normalizedRecords)
    } catch (error) {
      const detail = error?.response?.data?.detail
      setLoadError(detail || error.message || 'Unable to load class roster.')
      setRecords([])
    } finally {
      setIsLoading(false)
    }
  }, [queryParams])

  useEffect(() => {
    const timerId = window.setTimeout(() => {
      loadRoster()
    }, 0)

    return () => {
      window.clearTimeout(timerId)
    }
  }, [loadRoster])

  const onFilterChange = (event) => {
    const { name, value } = event.target
    setFilters((previous) => ({ ...previous, [name]: value }))
  }

  const onApplyFilters = (event) => {
    event.preventDefault()
    loadRoster()
  }

  return (
    <section className="surface-card">
      <header className="page-header">
        <div>
          <h2 className="page-title">Class Session Roster</h2>
          <p className="page-subtitle">
            Final attendance outcomes derived from aggregated sighting counts per student and class
            session date.
          </p>
        </div>

        <button type="button" className="ghost-btn" onClick={loadRoster} disabled={isLoading}>
          <RefreshCcw size={14} />
          <span>{isLoading ? 'Refreshing...' : 'Refresh'}</span>
        </button>
      </header>

      <form className="filters-grid" onSubmit={onApplyFilters}>
        <label>
          Course ID
          <input
            className="input-field"
            name="courseId"
            value={filters.courseId}
            onChange={onFilterChange}
            placeholder="Filter by course UUID"
          />
        </label>

        <label>
          Session Date
          <input
            className="input-field"
            type="date"
            name="sessionDate"
            value={filters.sessionDate}
            onChange={onFilterChange}
          />
        </label>

        <label>
          Status
          <select
            className="input-field"
            name="status"
            value={filters.status}
            onChange={onFilterChange}
          >
            <option value="all">All</option>
            <option value="present">Present</option>
            <option value="absent">Absent</option>
            <option value="late">Late</option>
            <option value="excused">Excused</option>
          </select>
        </label>

        <button type="submit" className="solid-btn" disabled={isLoading}>
          <Filter size={14} />
          <span>Apply Filters</span>
        </button>
      </form>

      {loadError ? (
        <div className="alert-banner" role="alert">
          <AlertCircle size={16} />
          <span>{loadError}</span>
        </div>
      ) : null}

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Student</th>
              <th>Course</th>
              <th>Session Date</th>
              <th>Sightings</th>
              <th>Required Sightings</th>
              <th>Status</th>
              <th>Evaluated At</th>
            </tr>
          </thead>

          <tbody>
            {records.length === 0 ? (
              <tr>
                <td colSpan={7}>
                  <div className="empty-state compact">
                    <p>No class-session attendance records found.</p>
                  </div>
                </td>
              </tr>
            ) : (
              records.map((record, index) => (
                <tr key={getRecordRowKey(record, index)}>
                  <td>{record.student_id}</td>
                  <td>{record.course_id}</td>
                  <td>{record.session_date || 'N/A'}</td>
                  <td>{record.sighting_count ?? 0}</td>
                  <td>{record.required_sightings ?? 0}</td>
                  <td>
                    <span className={`status-pill ${String(record.status).toLowerCase()}`}>
                      {mapStatusLabel(record.status)}
                    </span>
                  </td>
                  <td>
                    {record.evaluated_at
                      ? new Date(record.evaluated_at).toLocaleString()
                      : 'Not evaluated'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  )
}