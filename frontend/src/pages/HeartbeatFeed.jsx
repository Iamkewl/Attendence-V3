import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, RefreshCcw, Signal, SignalHigh, SignalLow } from 'lucide-react'
import client from '../api/client'

const MAX_EVENTS = 200
const RETRY_MAX_DELAY_MS = 30000
const RETRY_JITTER_MS = 400

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
    }
  } catch {
    return {
      eventId: createEventId(),
      channel: 'parse_error',
      eventType: 'unknown_payload',
      emittedAt: new Date().toISOString(),
      sighting: { raw: rawMessage },
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

export default function HeartbeatFeed() {
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
      if (!targetSocket) {
        return
      }

      targetSocket.onopen = null
      targetSocket.onmessage = null
      targetSocket.onerror = null
      targetSocket.onclose = null
    }

    const closeSocket = (targetSocket, code = 1000, reason = 'cleanup') => {
      if (!targetSocket) {
        return
      }

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

    const scheduleReconnect = (reason) => {
      if (cancelled) {
        return
      }

      attempts += 1
      setRetryAttempt(attempts)

      const baseDelayMs = 1000
      const exponentialDelayMs = Math.min(baseDelayMs * 2 ** (attempts - 1), RETRY_MAX_DELAY_MS)
      const jitterMs = Math.floor(Math.random() * RETRY_JITTER_MS)
      const reconnectDelayMs = exponentialDelayMs + jitterMs

      if (reason) {
        setLastError(reason)
      }
      setConnectionState('disconnected')

      clearRetryTimer()
      retryTimerId = window.setTimeout(() => {
        void connectSocket()
      }, reconnectDelayMs)
    }

    const connectSocket = async () => {
      if (cancelled) {
        return
      }

      clearRetryTimer()
      closeSocket(activeSocket, 1000, 'reconnecting')
      setConnectionState('connecting')
      connectionGeneration += 1
      const generation = connectionGeneration

      try {
        const ticketResponse = await client.post('/api/v1/auth/ws-ticket')

        if (cancelled || generation !== connectionGeneration) {
          return
        }

        const ticket = ticketResponse?.data?.ticket

        if (!ticket) {
          throw new Error('Realtime ticket response did not include a ticket.')
        }

        const nextSocket = new WebSocket(buildSocketUrl(ticket))
        activeSocket = nextSocket

        nextSocket.onopen = () => {
          if (cancelled || activeSocket !== nextSocket) {
            return
          }

          attempts = 0
          setRetryAttempt(0)
          setLastError('')
          setConnectionState('connected')
        }

        nextSocket.onmessage = (message) => {
          if (cancelled || activeSocket !== nextSocket) {
            return
          }

          const normalizedEvent = normalizeRealtimeMessage(message.data)
          setEvents((previous) => [normalizedEvent, ...previous].slice(0, MAX_EVENTS))
        }

        nextSocket.onerror = () => {
          if (cancelled || activeSocket !== nextSocket) {
            return
          }

          if (
            nextSocket.readyState === WebSocket.CONNECTING ||
            nextSocket.readyState === WebSocket.OPEN
          ) {
            nextSocket.close()
          }
        }

        nextSocket.onclose = (closeEvent) => {
          if (cancelled || activeSocket !== nextSocket) {
            return
          }

          activeSocket = null

          if (closeEvent.code === 1000) {
            setConnectionState('disconnected')
            return
          }

          const reason =
            closeEvent.reason || `WebSocket closed unexpectedly (code ${closeEvent.code}).`
          scheduleReconnect(reason)
        }
      } catch (error) {
        if (cancelled || generation !== connectionGeneration) {
          return
        }

        const apiMessage = error?.response?.data?.detail
        const reason = apiMessage || error.message || 'Failed to open realtime connection.'
        scheduleReconnect(reason)
      }
    }

    void connectSocket()

    return () => {
      cancelled = true
      clearRetryTimer()
      closeSocket(activeSocket, 1000, 'heartbeat-feed-unmounted')
    }
  }, [connectionKey])

  const connectionMeta = useMemo(() => {
    if (connectionState === 'connected') {
      return { label: 'Connected', tone: 'connected', icon: <SignalHigh size={14} /> }
    }

    if (connectionState === 'connecting') {
      return { label: 'Connecting', tone: 'connecting', icon: <Signal size={14} /> }
    }

    return { label: 'Disconnected', tone: 'disconnected', icon: <SignalLow size={14} /> }
  }, [connectionState])

  const reconnectNow = () => {
    setConnectionKey((value) => value + 1)
  }

  return (
    <section className="surface-card">
      <header className="page-header">
        <div>
          <h2 className="page-title">Live Sighting Feed</h2>
          <p className="page-subtitle">
            Streaming raw AI heartbeat sightings as camera batches arrive approximately every 10
            minutes.
          </p>
        </div>

        <div className={`connection-pill ${connectionMeta.tone}`}>
          {connectionMeta.icon}
          <span>{connectionMeta.label}</span>
        </div>
      </header>

      <div className="toolbar-row">
        <button type="button" className="ghost-btn" onClick={reconnectNow}>
          <RefreshCcw size={14} />
          <span>Reconnect</span>
        </button>

        {retryAttempt > 0 ? <p className="meta-note">Retry attempt: {retryAttempt}</p> : null}
      </div>

      {lastError ? (
        <div className="alert-banner" role="alert">
          <AlertCircle size={16} />
          <span>{lastError}</span>
        </div>
      ) : null}

      <div className="event-list" role="log" aria-live="polite">
        {events.length === 0 ? (
          <div className="empty-state">
            <p>No sightings received yet.</p>
            <small>The feed will populate when the next heartbeat batch is published.</small>
          </div>
        ) : (
          events.map((event) => (
            <article key={event.eventId} className="event-item">
              <header>
                <p className="event-title">{event.eventType}</p>
                <time dateTime={event.emittedAt}>{formatTimestamp(event.emittedAt)}</time>
              </header>

              <div className="event-grid">
                <p>
                  <strong>Channel:</strong> {event.channel}
                </p>
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
          ))
        )}
      </div>
    </section>
  )
}