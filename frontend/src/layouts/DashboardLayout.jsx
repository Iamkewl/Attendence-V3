import { ActivitySquare, ClipboardList, LayoutDashboard, LogOut, Menu, ScanFace, UserPlus, X } from 'lucide-react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useState } from 'react'
import { useAuth } from '../auth/AuthContext'

const RECOGNIZE_ROLES = ['admin', 'instructor']

const NAV_ITEMS = [
  {
    to: '/dashboard',
    label: 'Dashboard',
    icon: LayoutDashboard,
  },
  {
    to: '/heartbeat-feed',
    label: 'Live Feed',
    icon: ActivitySquare,
  },
  {
    to: '/roster',
    label: 'Class Roster',
    icon: ClipboardList,
  },
  {
    to: '/enroll',
    label: 'Enrollment',
    icon: UserPlus,
    roles: RECOGNIZE_ROLES,
  },
  {
    to: '/recognize',
    label: 'Recognize',
    icon: ScanFace,
    roles: RECOGNIZE_ROLES,
  },
]

const PAGE_TITLES = {
  '/dashboard': 'Dashboard',
  '/heartbeat-feed': 'Live Sighting Feed',
  '/roster': 'Class Roster',
  '/enroll': 'Enrollment',
  '/recognize': 'Face Recognition',
}

function getInitials(name, email) {
  if (name) {
    const parts = name.trim().split(' ')
    if (parts.length >= 2) return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
    return name.slice(0, 2).toUpperCase()
  }
  if (email) return email.slice(0, 2).toUpperCase()
  return 'U'
}

export default function DashboardLayout() {
  const { user, logout } = useAuth()
  const location = useLocation()
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false)

  const handleLogout = async () => {
    await logout()
  }

  const currentPageTitle = PAGE_TITLES[location.pathname] || 'Attendance V3'
  const userInitials = getInitials(user?.full_name, user?.email)
  const displayEmail = user?.email || user?.full_name || 'Unknown'
  const displayRole = user?.role || 'guest'

  return (
    <div className="dashboard-shell">
      {/* Sidebar */}
      <aside className={`sidebar ${isMobileSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="brand">
            <span className="brand-mark">A3</span>
            <div>
              <p className="brand-title">Attendance V3</p>
              <p className="brand-subtitle">Heartbeat Model</p>
            </div>
          </div>

          <button
            type="button"
            className="icon-btn mobile-only"
            aria-label="Close navigation"
            onClick={() => setIsMobileSidebarOpen(false)}
          >
            <X size={16} />
          </button>
        </div>

        <nav className="sidebar-nav" aria-label="Primary navigation">
          <p className="nav-section-label">Navigation</p>
          {NAV_ITEMS.filter((item) => !item.roles || item.roles.includes(user?.role)).map((item) => {
            const Icon = item.icon
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                onClick={() => setIsMobileSidebarOpen(false)}
              >
                <Icon size={16} className="nav-icon" aria-hidden="true" />
                <span>{item.label}</span>
              </NavLink>
            )
          })}
        </nav>

        <div className="sidebar-footer">
          <div className="user-chip" title={displayEmail}>
            <div className="user-chip-avatar" aria-hidden="true">{userInitials}</div>
            <div className="user-chip-info">
              <p className="user-chip-email">{displayEmail}</p>
              <p className="user-chip-role">{displayRole}</p>
            </div>
          </div>

          <button type="button" className="ghost-btn" onClick={handleLogout} style={{ width: '100%', justifyContent: 'center' }}>
            <LogOut size={14} aria-hidden="true" />
            <span>Sign Out</span>
          </button>
        </div>
      </aside>

      {/* Mobile backdrop */}
      {isMobileSidebarOpen ? (
        <button
          type="button"
          className="mobile-backdrop"
          aria-label="Close sidebar"
          onClick={() => setIsMobileSidebarOpen(false)}
        />
      ) : null}

      {/* Main content area */}
      <div className="content-shell">
        {/* Topbar */}
        <header className="topbar" role="banner">
          <div className="topbar-left">
            <button
              type="button"
              className="icon-btn mobile-only"
              aria-label="Open navigation"
              onClick={() => setIsMobileSidebarOpen(true)}
            >
              <Menu size={18} />
            </button>
            <span className="topbar-breadcrumb">{currentPageTitle}</span>
          </div>

          <div className="topbar-right">
            <div className="topbar-user-badge" aria-label={`Signed in as ${displayEmail}`}>
              <span className="topbar-user-email">{displayEmail}</span>
              <span className="topbar-role-pill">{displayRole}</span>
            </div>

            <button
              type="button"
              className="ghost-btn"
              onClick={handleLogout}
              aria-label="Sign out"
            >
              <LogOut size={14} aria-hidden="true" />
              <span>Sign Out</span>
            </button>
          </div>
        </header>

        <main className="content-main" id="main-content">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
