import { ActivitySquare, ClipboardList, LogOut, Menu, ShieldCheck, X } from 'lucide-react'
import { NavLink, Outlet } from 'react-router-dom'
import { useState } from 'react'
import { useAuth } from '../auth/AuthContext'

const navItems = [
  {
    to: '/heartbeat-feed',
    label: 'Heartbeat Feed',
    icon: ActivitySquare,
  },
  {
    to: '/class-roster',
    label: 'Class Roster',
    icon: ClipboardList,
  },
]

export default function DashboardLayout() {
  const { user, logout } = useAuth()
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false)

  const handleLogout = async () => {
    await logout()
  }

  return (
    <div className="dashboard-shell">
      <aside className={`sidebar ${isMobileSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="brand">
            <span className="brand-mark">A2</span>
            <div>
              <p className="brand-title">Attendance V2</p>
              <p className="brand-subtitle">Periodic Heartbeat Model</p>
            </div>
          </div>

          <button
            type="button"
            className="icon-btn mobile-only"
            aria-label="Close navigation"
            onClick={() => setIsMobileSidebarOpen(false)}
          >
            <X size={18} />
          </button>
        </div>

        <nav className="sidebar-nav" aria-label="Primary navigation">
          {navItems.map((item) => {
            const Icon = item.icon

            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                onClick={() => setIsMobileSidebarOpen(false)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </NavLink>
            )
          })}
        </nav>

        <div className="sidebar-footer">
          <div className="user-chip">
            <ShieldCheck size={16} />
            <div>
              <p>{user?.full_name || 'Unknown User'}</p>
              <small>{user?.role || 'guest'}</small>
            </div>
          </div>

          <button type="button" className="ghost-btn" onClick={handleLogout}>
            <LogOut size={16} />
            <span>Sign Out</span>
          </button>
        </div>
      </aside>

      {isMobileSidebarOpen ? (
        <button
          type="button"
          className="mobile-backdrop"
          aria-label="Close sidebar backdrop"
          onClick={() => setIsMobileSidebarOpen(false)}
        />
      ) : null}

      <div className="content-shell">
        <header className="topbar">
          <button
            type="button"
            className="icon-btn mobile-only"
            aria-label="Open navigation"
            onClick={() => setIsMobileSidebarOpen(true)}
          >
            <Menu size={18} />
          </button>

          <div>
            <h1 className="topbar-title">Live Attendance Operations</h1>
            <p className="topbar-subtitle">
              Monitor raw sightings and aggregated class-session attendance outcomes.
            </p>
          </div>
        </header>

        <main className="content-main">
          <Outlet />
        </main>
      </div>
    </div>
  )
}