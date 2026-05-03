import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { useAuth } from './auth/AuthContext'
import DashboardLayout from './layouts/DashboardLayout'
import HeartbeatFeed from './pages/HeartbeatFeed'
import ClassRoster from './pages/ClassRoster'
import LoginPage from './pages/LoginPage'

function SessionBootScreen() {
  return (
    <div className="auth-screen">
      <div className="auth-card compact">
        <h1>Attendance V2</h1>
        <p>Validating secure session cookies...</p>
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

function NotFoundPage() {
  return <Navigate to="/heartbeat-feed" replace />
}

function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />

      <Route
        path="/"
        element={
          <RequireAuth>
            <DashboardLayout />
          </RequireAuth>
        }
      >
        <Route index element={<Navigate to="/heartbeat-feed" replace />} />
        <Route path="heartbeat-feed" element={<HeartbeatFeed />} />
        <Route path="class-roster" element={<ClassRoster />} />
      </Route>

      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  )
}

export default App