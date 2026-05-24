import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, RefreshCcw, Signal, SignalHigh, SignalLow } from 'lucide-react'
import client from '../api/client'

const MAX_EVENTS = 200
// Backoff: 3s base, tripling each attempt (3/9/27/60s), cap 60s, jitter ±400ms
const RETRY_BASE_DELAY_MS = 3000
const RETRY_MAX_DELAY_MS = 60000
const RETRY_JITTER_MS = 400
// WebSocket close code 1008: Policy Violation (auth failure) — stop reconnecting
const WS_CLOSE_AUTH = 1008

const createEventId = () => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

const parseEventPayload = (rawPayload) => {
  if (typeof rawPayload === 'string') {
    try {
      return JSON.parse(rawPayload)
    } catch {
      return { raw: rawPayload }
    }
  }
  if (rawPayload && typeof rawPayload === 'object') {
    return rawPayload
  }
  return { raw: rawPayload }
}

const buildSocketUrl = (ticket) => {
  const apiUrl = import.meta.env.VITE_API_URL
  if (apiUrl) {
    const wsScheme = apiUrl.startsWith('https:') ? 'wss:' : 'ws:'
    const wsHost = apiUrl.replace(/^https?:/, '')
    return `${wsScheme}${wsHost}/ws/live?ticket=${encodeURIComponent(ticket)}`
  }
  const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${scheme}://${window.location.host}/ws/live?ticket=${encodeURIComponent(ticket)}`
}

const normalizeRealtimeMessage = (rawMessage) => {
  try {
    const frame = JSON.parse(rawMessage)
    const payload = parseEventPayload(frame?.payload)
    const sighting = payload?.sighting && typeof payload.sighting === 'object' ? payload.sighting : {}

    return {
      eventId: payload?.event_id || createEventId(),
      channel: frame?.channel || 'unknown',
      eventType: payload?.event_type || 'unknown',
      emittedAt: payload?.emitted_at || new Date().toISOString(),
      sighting,
      isNew: true,
    }
  } catch {
    return {
      eventId: createEventId(),
      channel: 'parse_error',
      eventType: 'unknown_payload',
      emittedAt: new Date().toISOString(),
      sighting: { raw: rawMessage },
      isNew: true,
    }
  }
}

const formatTimestamp = (value) => {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return 'N/A'
  }
  return parsed.toLocaleString()
}

function EventCard({ event }) {
  const cardRef = useRef(null)

  // Remove the new-event class after animation completes so it doesn't replay
  useEffect(() => {
    const el = cardRef.current
    if (!el) return
    const handleEnd = () => el.classList.remove('new-event')
    el.addEventListener('animationend', handleEnd, { once: true })
    return () => el.removeEventListener('animationend', handleEnd)
  }, [])

  return (
    <article
      ref={cardRef}
      className={`event-item${event.isNew ? ' new-event' : ''}`}
      aria-label={`Sighting event: ${event.eventType}`}
    >
      <header>
        <p className="event-title">{event.eventType}</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="event-channel-tag">{event.channel}</span>
          <time dateTime={event.emittedAt}>{formatTimestamp(event.emittedAt)}</time>
        </div>
      </header>

      <div className="event-grid">
        <p>
          <strong>Student:</strong> {event.sighting.student_id || 'Unknown'}
        </p>
        <p>
          <strong>Course:</strong> {event.sighting.course_id || 'Unknown'}
        </p>
        <p>
          <strong>Camera:</strong> {event.sighting.camera_id || 'Unknown'}
        </p>
        <p>
          <strong>Room:</strong> {event.sighting.room_id || 'N/A'}
        </p>
        <p>
          <strong>Confidence:</strong>{' '}
          {typeof event.sighting.confidence_score === 'number'
            ? `${Math.round(event.sighting.confidence_score * 100)}%`
            : 'N/A'}
        </p>
        <p>
          <strong>Sighting Time:</strong>{' '}
          {event.sighting.timestamp ? formatTimestamp(event.sighting.timestamp) : 'N/A'}
        </p>
      </div>
    </article>
  )
}

export default function HeartbeatFeed({ compact = false }) {
  const navigate = useNavigate()
  const [connectionState, setConnectionState] = useState('connecting')
  const [events, setEvents] = useState([])
  const [lastError, setLastError] = useState('')
  const [retryAttempt, setRetryAttempt] = useState(0)
  const [connectionKey, setConnectionKey] = useState(0)

  useEffect(() => {
    let cancelled = false
    let activeSocket = null
    let retryTimerId = null
    let connectionGeneration = 0
    let attempts = 0

    const clearRetryTimer = () => {
      if (retryTimerId !== null) {
        window.clearTimeout(retryTimerId)
        retryTimerId = null
      }
    }

    const detachSocketHandlers = (targetSocket) => {
      if (!targetSocket) return
      targetSocket.onopen = null
      targetSocket.onmessage = null
      targetSocket.onerror = null
      targetSocket.onclose = null
    }

    const closeSocket = (targetSocket, code = 1000, reason = 'cleanup') => {
      if (!targetSocket) return
      detachSocketHandlers(targetSocket)
      if (
        targetSocket.readyState === WebSocket.CONNECTING ||
        targetSocket.readyState === WebSocket.OPEN
      ) {
        targetSocket.close(code, reason)
      }
      if (activeSocket === targetSocket) {
        activeSocket = null
      }
    }

    const scheduleReconnect = (reason, authFailure = false) => {
      if (cancelled) return

      // Code 1008 means the server rejected the ticket as invalid/expired.
      if (authFailure) {
        setConnectionState('disconnected')
        setLastError('Session expired. Redirecting to login…')
        navigate('/login')
        return
      }

      attempts += 1
      setRetryAttempt(attempts)

      // Tripling backoff: 3s → 9s → 27s → 60s (capped), plus jitter
      const exponentialDelayMs = Math.min(
        RETRY_BASE_DELAY_MS * 3 ** (attempts - 1),
        RETRY_MAX_DELAY_MS,
      )
      const jitterMs = Math.floor(Math.random() * RETRY_JITTER_MS)
      const reconnectDelayMs = exponentialDelayMs + jitterMs

      if (reason) setLastError(reason)
      setConnectionState('reconnecting')

      console.info(
        `[HeartbeatFeed] Reconnect attempt ${attempts} scheduled in ${reconnectDelayMs}ms. Reason: ${reason || 'unknown'}`,
      )

      clearRetryTimer()
      retryTimerId = window.setTimeout(() => {
        void connectSocket()
      }, reconnectDelayMs)
    }

    const connectSocket = async () => {
      if (cancelled) return

      clearRetryTimer()
      closeSocket(activeSocket, 1000, 'reconnecting')
      setConnectionState('connecting')
      connectionGeneration += 1
      const generation = connectionGeneration

      try {
        const ticketResponse = await client.post('/api/v1/auth/ws-ticket')

        if (cancelled || generation !== connectionGeneration) return

        const ticket = ticketResponse?.data?.ticket

        if (!ticket) {
          throw new Error('Realtime ticket response did not include a ticket.')
        }

        const nextSocket = new WebSocket(buildSocketUrl(ticket))
        activeSocket = nextSocket

        nextSocket.onopen = () => {
          if (cancelled || activeSocket !== nextSocket) return
          attempts = 0
          setRetryAttempt(0)
          setLastError('')
          setConnectionState('connected')
        }

        nextSocket.onmessage = (message) => {
          if (cancelled || activeSocket !== nextSocket) return
          const normalizedEvent = normalizeRealtimeMessage(message.data)
          setEvents((previous) => [normalizedEvent, ...previous].slice(0, MAX_EVENTS))
        }

        nextSocket.onerror = () => {
          if (cancelled || activeSocket !== nextSocket) return
          if (
            nextSocket.readyState === WebSocket.CONNECTING ||
            nextSocket.readyState === WebSocket.OPEN
          ) {
            nextSocket.close()
          }
        }

        nextSocket.onclose = (closeEvent) => {
          if (cancelled || activeSocket !== nextSocket) return
          activeSocket = null
          if (closeEvent.code === 1000) {
            setConnectionState('disconnected')
            return
          }
          const isAuthFailure = closeEvent.code === WS_CLOSE_AUTH
          const reason =
            closeEvent.reason || `WebSocket closed unexpectedly (code ${closeEvent.code}).`
          scheduleReconnect(reason, isAuthFailure)
        }
      } catch (error) {
        if (cancelled || generation !== connectionGeneration) return
        const statusCode = error?.response?.status
        const isAuthFailure = statusCode === 401
        const apiMessage = error?.response?.data?.detail
        const reason = apiMessage || error.message || 'Failed to open realtime connection.'
        scheduleReconnect(reason, isAuthFailure)
      }
    }

    void connectSocket()

    return () => {
      cancelled = true
      clearRetryTimer()
      closeSocket(activeSocket, 1000, 'heartbeat-feed-unmounted')
    }
  }, [connectionKey, navigate])

  const connectionMeta = useMemo(() => {
    if (connectionState === 'connected') {
      return {
        label: 'Connected',
        tone: 'connected',
        icon: <SignalHigh size={13} aria-hidden="true" />,
        pulse: true,
      }
    }
    if (connectionState === 'connecting') {
      return {
        label: 'Connecting…',
        tone: 'connecting',
        icon: <Signal size={13} aria-hidden="true" />,
        pulse: false,
      }
    }
    if (connectionState === 'reconnecting') {
      return {
        label: `Reconnecting (${retryAttempt})`,
        tone: 'connecting',
        icon: <Signal size={13} aria-hidden="true" />,
        pulse: false,
      }
    }
    return {
      label: 'Disconnected',
      tone: 'disconnected',
      icon: <SignalLow size={13} aria-hidden="true" />,
      pulse: false,
    }
  }, [connectionState, retryAttempt])

  const reconnectNow = () => {
    setConnectionKey((value) => value + 1)
  }

  return (
    <section className="surface-card" aria-labelledby="hb-feed-title">
      <header className="page-header">
        <div>
          <h2 className="page-title" id="hb-feed-title">
            {compact ? 'Live Feed' : 'Live Sighting Feed'}
          </h2>
          {!compact && (
            <p className="page-subtitle">
              Streaming raw AI heartbeat sightings as camera batches arrive approximately every 10
              minutes.
            </p>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0 }}>
          <div
            className={`connection-pill ${connectionMeta.tone}`}
            role="status"
            aria-label={`Connection status: ${connectionMeta.label}`}
          >
            <span className={`connection-dot${connectionMeta.pulse ? ' pulse' : ''}`} aria-hidden="true" />
            {connectionMeta.icon}
            <span>{connectionMeta.label}</span>
          </div>

          <button
            type="button"
            className="ghost-btn"
            onClick={reconnectNow}
            aria-label="Force reconnect"
          >
            <RefreshCcw size={13} aria-hidden="true" />
            {!compact && <span>Reconnect</span>}
          </button>
        </div>
      </header>

      {lastError ? (
        <div className="alert-banner" role="alert">
          <AlertCircle size={15} aria-hidden="true" />
          <span>{lastError}</span>
        </div>
      ) : null}

      <div
        className="event-list"
        role="log"
        aria-live="polite"
        aria-label="Live sighting events"
        style={compact ? { maxHeight: '420px' } : {}}
      >
        {events.length === 0 ? (
          <div className="empty-state">
            <Signal size={28} style={{ color: 'var(--text-subtle)', marginBottom: '8px' }} aria-hidden="true" />
            <p>No sightings received yet.</p>
            <small>The feed will populate when the next heartbeat batch is published.</small>
          </div>
        ) : (
          events.map((event) => (
            <EventCard key={event.eventId} event={event} />
          ))
        )}
      </div>
    </section>
  )
}
