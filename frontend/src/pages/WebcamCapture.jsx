import { useCallback, useEffect, useRef, useState } from 'react'
import { Camera, RotateCcw } from 'lucide-react'

/**
 * WebcamCapture
 *
 * Props:
 *   onCapture(blob: Blob) — called with a JPEG blob when the user snapshots
 *   capturedBlob: Blob | null — if set, shows preview + Retake/Submit controls
 *   onRetake() — resets to live view
 *   onSubmit() — confirms the captured frame for submission
 *   isSubmitting: boolean
 */
export default function WebcamCapture({ onCapture, capturedBlob, onRetake, onSubmit, isSubmitting }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [capturedUrl, setCapturedUrl] = useState(null)

  // Revoke old object URL to avoid memory leak when blob changes.
  // setCapturedUrl calls are intentional: this effect manages a derived resource
  // (an object URL) that must be created and cleaned up reactively from a prop.
  useEffect(() => {
    if (capturedBlob) {
      const url = URL.createObjectURL(capturedBlob)
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setCapturedUrl(url)
      return () => URL.revokeObjectURL(url)
    } else {
      setCapturedUrl(null)
    }
  }, [capturedBlob])

  // Stop camera track when a snapshot is confirmed
  useEffect(() => {
    if (capturedBlob && streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      setIsCameraActive(false)
    }
  }, [capturedBlob])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  // Attach stream once <video> mounts — isCameraActive gates its render so
  // videoRef.current is null until after setIsCameraActive(true) flushes.
  useEffect(() => {
    if (isCameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
      videoRef.current.play().catch(() => {})
    }
  }, [isCameraActive])

  const startCamera = useCallback(async () => {
    setCameraError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      })
      streamRef.current = stream
      setIsCameraActive(true) // mounts <video>, then the effect above wires srcObject
    } catch (err) {
      const name = err?.name || ''
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setCameraError('Camera permission was denied. Please allow camera access and try again.')
      } else if (name === 'NotFoundError') {
        setCameraError('No camera detected on this device.')
      } else {
        setCameraError(err.message || 'Unable to access camera.')
      }
    }
  }, [])

  const takeSnapshot = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    canvas.toBlob(
      (blob) => {
        if (blob) onCapture(blob)
      },
      'image/jpeg',
      0.92,
    )
  }, [onCapture])

  const handleRetake = useCallback(() => {
    onRetake()
    startCamera()
  }, [onRetake, startCamera])

  // Snapshot preview + controls
  if (capturedBlob && capturedUrl) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center' }}>
        <img
          src={capturedUrl}
          alt="Captured enrollment frame — review before submitting"
          style={{
            width: '100%',
            maxWidth: 640,
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border)',
            display: 'block',
          }}
        />
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', justifyContent: 'center' }}>
          <button
            type="button"
            className="ghost-btn"
            onClick={handleRetake}
            disabled={isSubmitting}
          >
            <RotateCcw size={15} aria-hidden="true" />
            <span>Retake</span>
          </button>
          <button
            type="button"
            className="solid-btn"
            onClick={onSubmit}
            disabled={isSubmitting}
          >
            <Camera size={15} aria-hidden="true" />
            <span>{isSubmitting ? 'Submitting...' : 'Submit Enrollment'}</span>
          </button>
        </div>
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>
    )
  }

  // Live camera feed
  if (isCameraActive) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center' }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          aria-label="Live camera preview for enrollment"
          style={{
            width: '100%',
            maxWidth: 640,
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border)',
            background: '#000',
            display: 'block',
          }}
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        <button
          type="button"
          className="solid-btn"
          onClick={takeSnapshot}
          onKeyDown={(e) => e.key === 'Enter' && takeSnapshot()}
        >
          <Camera size={15} aria-hidden="true" />
          <span>Take Snapshot</span>
        </button>
      </div>
    )
  }

  // Initial idle state — camera not yet requested
  return (
    <div
      className="empty-state"
      style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}
    >
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <Camera size={40} style={{ color: 'var(--text-subtle)' }} aria-hidden="true" />
      <p style={{ fontWeight: 600 }}>Camera not started</p>
      <small>Click below to request camera access — permissions are not requested until you do.</small>
      {cameraError ? (
        <div className="alert-banner" role="alert" style={{ width: '100%', marginTop: 4 }}>
          <span>{cameraError}</span>
        </div>
      ) : null}
      <button
        type="button"
        className="solid-btn"
        style={{ marginTop: 6 }}
        onClick={startCamera}
        aria-label="Enable webcam for face capture"
      >
        <Camera size={15} aria-hidden="true" />
        <span>Enable Camera</span>
      </button>
    </div>
  )
}
