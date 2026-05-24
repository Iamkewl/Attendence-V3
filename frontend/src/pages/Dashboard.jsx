import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ActivitySquare, ClipboardList, RefreshCcw, Users } from 'lucide-react'
import client from '../api/client'
import HeartbeatFeed from './HeartbeatFeed'

const ATTENDANCE_SESSIONS_ENDPOINT = '/api/v1/attendance/sessions'

const DEFAULT_COURSE = ''

function StatCard({ label, value, variant, sub }) {
  return (
    <div className="stat-card">
      <p className="stat-card-label">{label}</p>
      <p className={`stat-card-value ${variant || ''}`}>{value}</p>
      {sub ? <p className="stat-card-sub">{sub}</p> : null}
    </div>
  )
}

function ActivityItem({ dot, text, time }) {
  return (
    <div className="activity-item">
      <span className={`activity-dot ${dot}`} aria-hidden="true" />
      <div className="activity-info">
        <p>{text}</p>
        <small>{time}</small>
      </div>
    </div>
  )
}

function formatRelative(dateStr) {
  if (!dateStr) return ''
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return new Date(dateStr).toLocaleDateString()
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [stats, setStats] = useState({ present: 0, absent: 0, pending: 0, total: 0 })
  const [recentActivity, setRecentActivity] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const loadStats = useCallback(async () => {
    setIsLoading(true)
    try {
      const today = new Date().toISOString().split('T')[0]
      const response = await client.get(ATTENDANCE_SESSIONS_ENDPOINT, {
        params: { session_date: today, course_id: DEFAULT_COURSE || undefined },
      })
      const payload = response.data
      const records = Array.isArray(payload?.records) ? payload.records : []

      const counts = { present: 0, absent: 0, pending: 0, total: records.length }
      const activity = []

      for (const rec of records) {
        const status = String(rec?.status ?? '').toLowerCase()
        if (status === 'present') counts.present++
        else if (status === 'absent') counts.absent++
        else counts.pending++

        activity.push({
          id: rec.id || rec.student_id,
          dot: status === 'present' ? 'present' : status === 'absent' ? 'absent' : 'pending',
          text: `${rec.student_full_name || rec.student_id || 'Student'} marked ${status || 'pending'}`,
          time: formatRelative(rec.evaluated_at || rec.updated_at),
        })
      }

      // Sort activity: most recently evaluated first
      activity.sort((a, b) => {
        const aRec = records.find((r) => (r.id || r.student_id) === a.id) || {}
        const bRec = records.find((r) => (r.id || r.student_id) === b.id) || {}
        return (
          new Date(bRec.evaluated_at || bRec.updated_at || 0).getTime() -
          new Date(aRec.evaluated_at || aRec.updated_at || 0).getTime()
        )
      })

      setStats(counts)
      setRecentActivity(activity.slice(0, 10))
    } catch (error) {
      const statusCode = error?.response?.status
      if (statusCode === 401) {
        navigate('/login')
      }
      // 404 or no data: leave zeroes — not an error state on the dashboard
    } finally {
      setIsLoading(false)
    }
  }, [navigate])

  useEffect(() => {
    // loadStats is async; all setState calls after the first await are async
    // callbacks. The synchronous setIsLoading(true) is an intentional loading
    // indicator — not a cascading render concern.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadStats()
  }, [loadStats])

  return (
    <>
      {/* Stat row */}
      <div className="surface-card">
        <div className="page-header">
          <div>
            <h2 className="page-title">Today's Attendance Overview</h2>
            <p className="page-subtitle">
              Aggregated outcomes for today's sessions across all enrolled students.
            </p>
          </div>
          <button
            type="button"
            className="ghost-btn"
            onClick={loadStats}
            disabled={isLoading}
            aria-label="Refresh stats"
          >
            <RefreshCcw size={14} aria-hidden="true" />
            <span>{isLoading ? 'Loading...' : 'Refresh'}</span>
          </button>
        </div>

        <div className="stat-card-row">
          <StatCard
            label="Present"
            value={stats.present}
            variant="present"
            sub={stats.total > 0 ? `${Math.round((stats.present / stats.total) * 100)}% of total` : 'No data yet'}
          />
          <StatCard
            label="Absent"
            value={stats.absent}
            variant="absent"
            sub={stats.total > 0 ? `${Math.round((stats.absent / stats.total) * 100)}% of total` : 'No data yet'}
          />
          <StatCard
            label="Pending"
            value={stats.pending}
            variant="pending"
            sub="Awaiting evaluation"
          />
          <StatCard
            label="Total Students"
            value={stats.total}
            variant=""
            sub="Enrolled in session"
          />
        </div>
      </div>

      {/* Two-column grid: feed + recent activity */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 340px', gap: '20px', alignItems: 'start' }}>
        <HeartbeatFeed compact />

        {/* Recent Activity Panel */}
        <div className="surface-card" style={{ position: 'sticky', top: '76px' }}>
          <div className="page-header" style={{ marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Users size={16} style={{ color: 'var(--text-subtle)' }} aria-hidden="true" />
              <h2 className="page-title" style={{ fontSize: '0.95rem' }}>Recent Activity</h2>
            </div>
            <a
              href="/roster"
              className="ghost-btn"
              style={{ fontSize: '0.78rem', padding: '5px 10px' }}
              aria-label="View full roster"
            >
              <ClipboardList size={13} aria-hidden="true" />
              <span>Full Roster</span>
            </a>
          </div>

          {recentActivity.length === 0 ? (
            <div className="empty-state compact" style={{ borderStyle: 'dashed' }}>
              <ActivitySquare size={24} style={{ color: 'var(--text-subtle)', marginBottom: '6px' }} aria-hidden="true" />
              <p>No activity yet today</p>
              <small>Records will appear once evaluation runs.</small>
            </div>
          ) : (
            <div className="activity-list" role="list">
              {recentActivity.map((item, idx) => (
                <ActivityItem key={idx} dot={item.dot} text={item.text} time={item.time} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Responsive: stack on narrow viewports */}
      <style>{`
        @media (max-width: 900px) {
          .dashboard-two-col { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </>
  )
}
