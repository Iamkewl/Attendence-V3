import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { useAuth } from './auth/AuthContext'
import DashboardLayout from './layouts/DashboardLayout'
import Dashboard from './pages/Dashboard'
import HeartbeatFeed from './pages/HeartbeatFeed'
import ClassRoster from './pages/ClassRoster'
import LoginPage from './pages/LoginPage'
import Enrollment from './pages/Enrollment'
import Recognize from './pages/Recognize'

const RECOGNIZE_ROLES = ['admin', 'instructor']

function SessionBootScreen() {
  return (
    <div className="auth-screen">
      <div className="auth-card compact">
        <div className="auth-logo-mark" aria-hidden="true">A3</div>
        <h1>Attendance V3</h1>
        <p>Validating session…</p>
      </div>
    </div>
  )
}

function RequireAuth({ children }) {
  const location = useLocation()
  const { isAuthenticated, isLoading } = useAuth()

  if (isLoading) {
    return <SessionBootScreen />
  }

  if (!isAuthenticated) {
    const fromPath = `${location.pathname}${location.search}`
    return <Navigate to="/login" replace state={{ from: fromPath }} />
  }

  return children
}

function RequireRole({ roles, children }) {
  const { user } = useAuth()
  if (!roles.includes(user?.role)) {
    return <Navigate to="/dashboard" replace />
  }
  return children
}

function NotFoundPage() {
  return <Navigate to="/dashboard" replace />
}

function App() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/login" element={<LoginPage />} />

      {/* Authenticated shell */}
      <Route
        path="/"
        element={
          <RequireAuth>
            <DashboardLayout />
          </RequireAuth>
        }
      >
        {/* Default: / → /dashboard */}
        <Route index element={<Navigate to="/dashboard" replace />} />

        {/* web-ui-designer owns these routes */}
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="heartbeat-feed" element={<HeartbeatFeed />} />
        <Route path="roster" element={<ClassRoster />} />

        <Route path="enroll" element={<Enrollment />} />

        {/* Admin / Instructor only */}
        <Route
          path="recognize"
          element={
            <RequireRole roles={RECOGNIZE_ROLES}>
              <Recognize />
            </RequireRole>
          }
        />
      </Route>

      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  )
}

export default App
