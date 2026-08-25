import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, Camera, ImagePlus, RefreshCcw, ScanFace, UploadCloud } from 'lucide-react'
import client from '../api/client'
import { resolveError } from './recognizeErrors'
import AnnotatedImage from '../components/AnnotatedImage'

const RECOGNIZE_ENDPOINT = '/api/v1/inference/photo'
const BATCH_ENDPOINT = '/api/v1/inference/batch'
const MAX_FILE_BYTES = 10 * 1024 * 1024 // 10 MB

// Classroom burst capture (multi-frame tracking path)
const BURST_FRAME_COUNT = 5
const BURST_FRAME_INTERVAL_MS = 400
const MAX_TENSOR_SIDE_PX = 640
const TASK_POLL_INTERVAL_MS = 1500
const TASK_POLL_MAX_ATTEMPTS = 40

function formatBytes(n) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function uint8ArrayToBase64(bytes) {
  let binary = ''
  const chunkSize = 0x8000
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(offset, offset + chunkSize))
  }
  return btoa(binary)
}

function describeFailure(err) {
  const status = err?.response?.status
  if (status) return `HTTP ${status}`
  return err?.message ? 'network error' : 'unknown error'
}

// Map the async batch task result onto the photo-response shape already
// rendered by StatsRow / AnnotatedImage / ResultsTable. Rows stay keyed by
// track_id — the server guarantees unique track ids within one batch.
function batchResultToPhotoShape(batchResult, frameWidth, frameHeight) {
  const rows = Array.isArray(batchResult?.results) ? batchResult.results : []
  const detections = rows.map((row) => ({
    track_id: row.track_id,
    bbox: row.bbox,
    confidence: row.detection_score,
    liveness_score: row.liveness_score,
    is_live: row.is_live,
    match:
      row.is_match && row.student_id
        ? {
            // The batch/task-status API reports student IDs, not names.
            student_full_name: '',
            student_number: '',
            student_id: row.student_id,
            cosine_similarity: row.cosine_similarity,
          }
        : null,
  }))
  return {
    image_width: frameWidth,
    image_height: frameHeight,
    detection_count: batchResult?.detection_count ?? detections.length,
    match_count: detections.filter((det) => det.match).length,
    processed_at: batchResult?.generated_at,
    detections,
  }
}

// ── Spinner ────────────────────────────────────────────────────────────────
function Spinner({ label }) {
  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 14,
        padding: '48px 0',
      }}
    >
      <div
        aria-label={label}
        style={{
          width: 44,
          height: 44,
          border: '4px solid rgba(79,70,229,0.15)',
          borderTopColor: 'var(--accent)',
          borderRadius: '50%',
          animation: 'spin 0.75s linear infinite',
        }}
      />
      <p style={{ color: 'var(--text-soft)', fontWeight: 600, margin: 0, fontSize: '0.9rem' }}>
        {label}
      </p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

// ── Stats Row ──────────────────────────────────────────────────────────────
function StatsRow({ detectionCount, matchCount }) {
  const unknown = detectionCount - matchCount
  return (
    <div
      style={{
        display: 'flex',
        gap: 20,
        flexWrap: 'wrap',
        padding: '10px 14px',
        background: 'var(--surface-muted)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-md)',
        fontSize: '0.875rem',
        fontWeight: 600,
        color: 'var(--text-soft)',
      }}
      aria-label="Recognition statistics"
    >
      <span>
        Detected: <strong style={{ color: 'var(--text-main)' }}>{detectionCount}</strong>
      </span>
      <span style={{ color: 'var(--border-strong)' }}>|</span>
      <span>
        Matched:{' '}
        <strong style={{ color: 'var(--present-text)' }}>{matchCount}</strong>
      </span>
      <span style={{ color: 'var(--border-strong)' }}>|</span>
      <span>
        Unknown:{' '}
        <strong style={{ color: 'var(--pending-text)' }}>{unknown}</strong>
      </span>
    </div>
  )
}

// ── Results Table ──────────────────────────────────────────────────────────
function ResultsTable({ detections }) {
  if (!detections || detections.length === 0) return null

  return (
    <div style={{ marginTop: 20 }}>
      <h3
        style={{
          fontSize: '0.9rem',
          fontWeight: 700,
          color: 'var(--text-main)',
          marginBottom: 10,
        }}
      >
        Detection Details
      </h3>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Track</th>
              <th>BBox (x,y,w,h)</th>
              <th>Det. Conf</th>
              <th>Liveness</th>
              <th>Live?</th>
              <th>Match Name</th>
              <th>Student No.</th>
              <th>Similarity</th>
            </tr>
          </thead>
          <tbody>
            {detections.map((det) => (
              <tr key={det.track_id}>
                <td style={{ fontVariantNumeric: 'tabular-nums' }}>{det.track_id}</td>
                <td style={{ fontFamily: 'monospace', fontSize: '0.78rem' }}>
                  {det.bbox.map((v) => Math.round(v)).join(', ')}
                </td>
                <td>{(det.confidence * 100).toFixed(1)}%</td>
                <td>{(det.liveness_score * 100).toFixed(1)}%</td>
                <td>
                  <span
                    className={`status-pill ${det.is_live ? 'present' : 'absent'}`}
                  >
                    {det.is_live ? 'Yes' : 'No'}
                  </span>
                </td>
                <td style={{ fontWeight: det.match ? 600 : 400, color: det.match ? 'var(--present-text)' : 'var(--text-subtle)' }}>
                  {det.match ? det.match.student_full_name || det.match.student_id || '—' : '—'}
                </td>
                <td style={{ fontFamily: 'monospace', fontSize: '0.82rem' }}>
                  {det.match ? det.match.student_number || '—' : '—'}
                </td>
                <td>
                  {det.match?.cosine_similarity != null
                    ? `${(det.match.cosine_similarity * 100).toFixed(1)}%`
                    : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Drop Zone ──────────────────────────────────────────────────────────────
function DropZone({ onFile }) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)

  const handleFiles = useCallback(
    (files) => {
      const file = files[0]
      if (!file) return
      if (!file.type.startsWith('image/')) {
        return
      }
      onFile(file)
    },
    [onFile],
  )

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setIsDragging(false)
      handleFiles(e.dataTransfer.files)
    },
    [handleFiles],
  )

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Drop zone: drag and drop an image or click to browse"
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click() }}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 12,
        padding: '48px 24px',
        border: `2px dashed ${isDragging ? 'var(--accent)' : 'var(--border-strong)'}`,
        borderRadius: 'var(--radius-lg)',
        background: isDragging ? 'var(--accent-soft)' : 'var(--surface-muted)',
        cursor: 'pointer',
        transition: 'border-color 0.15s, background 0.15s',
        outline: 'none',
      }}
    >
      <UploadCloud
        size={36}
        style={{ color: isDragging ? 'var(--accent)' : 'var(--text-subtle)' }}
        aria-hidden="true"
      />
      <div style={{ textAlign: 'center' }}>
        <p style={{ fontWeight: 600, color: 'var(--text-soft)', margin: 0, fontSize: '0.9rem' }}>
          Drag &amp; drop a photo here
        </p>
        <p style={{ margin: '4px 0 0', fontSize: '0.82rem', color: 'var(--text-subtle)' }}>
          or click to browse — JPEG / PNG, max 10 MB
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png"
        style={{ display: 'none' }}
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  )
}

// ── Burst Capture Panel ────────────────────────────────────────────────────
function BurstCapturePanel({ videoRef, isActive, error, busy, onStartCamera, onBurst }) {
  if (!isActive) {
    return (
      <div
        className="empty-state"
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}
      >
        <Camera size={40} style={{ color: 'var(--text-subtle)' }} aria-hidden="true" />
        <p style={{ fontWeight: 600 }}>Camera not started</p>
        <small>
          Burst mode captures {BURST_FRAME_COUNT} frames ~{BURST_FRAME_INTERVAL_MS} ms apart from
          this device&apos;s webcam and runs them through multi-frame tracking.
        </small>
        {error && (
          <div className="alert-banner" role="alert" style={{ width: '100%', marginTop: 4 }}>
            <span>{error}</span>
          </div>
        )}
        <button
          type="button"
          className="solid-btn"
          style={{ marginTop: 6 }}
          onClick={onStartCamera}
          aria-label="Enable webcam for classroom burst capture"
        >
          <Camera size={15} aria-hidden="true" />
          <span>Enable Camera</span>
        </button>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center' }}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        aria-label="Live camera preview for classroom burst capture"
        style={{
          width: '100%',
          maxWidth: 640,
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border)',
          background: '#000',
          display: 'block',
        }}
      />
      <button type="button" className="solid-btn" onClick={onBurst} disabled={busy}>
        <ScanFace size={15} aria-hidden="true" />
        <span>{busy ? 'Capturing…' : `Capture ${BURST_FRAME_COUNT}-Frame Burst`}</span>
      </button>
    </div>
  )
}

// ── Main Page ──────────────────────────────────────────────────────────────
export default function Recognize() {
  const navigate = useNavigate()

  // 'idle' | 'preview' | 'loading' | 'result'
  const [stage, setStage] = useState('idle')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  // Classroom burst (multi-frame) mode
  const [burstMode, setBurstMode] = useState(false)
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [burstBusy, setBurstBusy] = useState(false)
  const [notice, setNotice] = useState('')

  const handleFile = useCallback((file) => {
    if (file.size > MAX_FILE_BYTES) {
      setError('File exceeds 10 MB. Please choose a smaller image.')
      return
    }
    setError('')
    setSelectedFile(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreviewUrl(e.target.result)
      setStage('preview')
    }
    reader.readAsDataURL(file)
  }, [])

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    setCameraActive(false)
  }, [])

  const startCamera = useCallback(async () => {
    setCameraError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      })
      streamRef.current = stream
      setCameraActive(true)
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

  // Wire the MediaStream once <video> mounts; release the camera on unmount.
  useEffect(() => {
    if (cameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
      videoRef.current.play().catch(() => {})
    }
  }, [cameraActive])

  useEffect(
    () => () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
      }
    },
    [],
  )

  // Grab K frames ~400 ms apart from the live webcam, downscaled so the
  // longest side is capped at MAX_TENSOR_SIDE_PX (aspect preserved). Each
  // canvas readback yields RGBA via getImageData; the alpha channel is
  // stripped into a packed RGB uint8 tensor.
  const captureBurstFrames = useCallback(async () => {
    const video = videoRef.current
    if (!video || !video.videoWidth || !video.videoHeight) {
      throw new Error('Camera frame is not ready yet. Wait for the preview and try again.')
    }

    const scale = Math.min(1, MAX_TENSOR_SIDE_PX / Math.max(video.videoWidth, video.videoHeight))
    const width = Math.max(1, Math.round(video.videoWidth * scale))
    const height = Math.max(1, Math.round(video.videoHeight * scale))

    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d', { willReadFrequently: true })

    let previewDataUrl = ''
    const captured = []
    for (let index = 0; index < BURST_FRAME_COUNT; index += 1) {
      if (index > 0) await sleep(BURST_FRAME_INTERVAL_MS)
      ctx.drawImage(video, 0, 0, width, height)
      if (index === 0) previewDataUrl = canvas.toDataURL('image/jpeg', 0.9)

      const rgba = ctx.getImageData(0, 0, width, height).data
      const rgb = new Uint8Array(width * height * 3)
      for (let src = 0, dst = 0; src < rgba.length; src += 4) {
        rgb[dst] = rgba[src]
        rgb[dst + 1] = rgba[src + 1]
        rgb[dst + 2] = rgba[src + 2]
        dst += 3
      }
      captured.push({ rgb, width, height })
    }

    return { frames: captured, canvas, previewUrl: previewDataUrl }
  }, [])

  const pollBatchTask = useCallback(async (taskId) => {
    for (let attempt = 0; attempt < TASK_POLL_MAX_ATTEMPTS; attempt += 1) {
      await sleep(TASK_POLL_INTERVAL_MS)
      const response = await client.get(`/api/v1/inference/tasks/${taskId}`)
      const { state, result: taskResult, error } = response.data ?? {}
      if (state === 'SUCCESS') return taskResult
      if (state === 'FAILURE' || state === 'REVOKED') {
        throw new Error(error || 'Multi-frame inference task failed.')
      }
    }
    throw new Error('Timed out waiting for multi-frame inference results.')
  }, [])

  const handleRecognize = useCallback(async () => {
    if (!selectedFile) return
    setStage('loading')
    setError('')

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      // ATT-034: do not set `Content-Type: 'multipart/form-data'` manually.
      // Doing so strips the `boundary=...` parameter that axios computes
      // from the FormData body; python-multipart (FastAPI/Starlette) then
      // rejects the request with "no boundary found" / 422. Letting axios
      // infer the header means the request is sent with
      // `multipart/form-data; boundary=...` as required.
      const response = await client.post(RECOGNIZE_ENDPOINT, formData, {
        // ATT-034: never send a hand-rolled multipart Content-Type (it lacks
        // the boundary). Also strip the JSON default from the shared axios
        // instance — with a JSON content type axios serializes FormData into
        // `{"file":{}}` and silently drops the file.
        headers: { 'Content-Type': undefined },
        timeout: 60000,
      })
      setResult(response.data)
      setStage('result')
    } catch (err) {
      const msg = resolveError(err)
      if (msg === '__REDIRECT_LOGIN__') {
        navigate('/login')
        return
      }
      setError(msg)
      setStage('preview')
    }
  }, [selectedFile, navigate])

  const handleBurstSubmit = useCallback(async () => {
    setBurstBusy(true)
    setError('')
    setNotice('')

    let burstCanvas = null
    try {
      const { frames, canvas, previewUrl: framePreview } = await captureBurstFrames()
      burstCanvas = canvas
      setPreviewUrl(framePreview)
      setStage('loading')

      const payload = {
        frames: frames.map((frame, index) => ({
          frame_id: `kiosk-burst-${Date.now()}-${index}`,
          data_base64: uint8ArrayToBase64(frame.rgb),
          width: frame.width,
          height: frame.height,
          channels: 3,
          dtype: 'uint8',
          normalize: true,
        })),
      }

      const response = await client.post(BATCH_ENDPOINT, payload, { timeout: 30000 })
      if (response.status !== 202) {
        throw new Error(`Batch endpoint returned unexpected status ${response.status}.`)
      }

      const taskId = response.data?.task_id
      if (!taskId) throw new Error('Batch endpoint did not return a task id.')

      const batchResult = await pollBatchTask(taskId)
      setResult(batchResultToPhotoShape(batchResult, frames[0].width, frames[0].height))
      setPreviewUrl(framePreview)
      setStage('result')
    } catch (batchErr) {
      // Batch path failed (non-202, network, or failed task): fall back to the
      // existing single-frame photo flow with a captured frame and say so.
      if (!burstCanvas) {
        const msg =
          batchErr?.message || 'Burst capture failed before any frame could be read from the camera.'
        setError(msg)
        setStage('idle')
        return
      }
      try {
        const blob = await new Promise((resolve, reject) => {
          burstCanvas.toBlob(
            (b) => (b ? resolve(b) : reject(new Error('Could not encode the captured frame.'))),
            'image/jpeg',
            0.92,
          )
        })
        const formData = new FormData()
        formData.append('file', blob, 'kiosk-burst-frame.jpg')
        const response = await client.post(RECOGNIZE_ENDPOINT, formData, {
          headers: { 'Content-Type': undefined }, // ATT-034 — see handleRecognize
          timeout: 60000,
        })
        setNotice(
          `Multi-frame batch unavailable (${describeFailure(batchErr)}) — showing a single-frame result instead.`,
        )
        setResult(response.data)
        setPreviewUrl(burstCanvas.toDataURL('image/jpeg', 0.92))
        setStage('result')
      } catch (fallbackErr) {
        const msg = resolveError(fallbackErr)
        if (msg === '__REDIRECT_LOGIN__') {
          navigate('/login')
          return
        }
        setError(msg)
        setNotice(`Multi-frame batch unavailable (${describeFailure(batchErr)}).`)
        setStage('idle')
      }
    } finally {
      setBurstBusy(false)
    }
  }, [captureBurstFrames, pollBatchTask, navigate])

  const handleBurstToggle = useCallback(
    (event) => {
      const enabled = event.target.checked
      setBurstMode(enabled)
      setError('')
      setNotice('')
      setCameraError('')
      if (!enabled) stopCamera()
    },
    [stopCamera],
  )

  const handleReset = useCallback(() => {
    setStage('idle')
    setSelectedFile(null)
    setPreviewUrl(null)
    setError('')
    setResult(null)
    setNotice('')
    if (burstMode) stopCamera()
  }, [burstMode, stopCamera])

  return (
    <section className="surface-card">
      {/* Page header */}
      <header className="page-header">
        <div>
          <h2 className="page-title">Face Recognition</h2>
          <p className="page-subtitle">
            Upload a group photo to run detection and recognition. Matched students and
            liveness results appear as overlaid bounding boxes.
          </p>
        </div>
        {stage === 'result' && (
          <button type="button" className="ghost-btn" onClick={handleReset}>
            <RefreshCcw size={14} aria-hidden="true" />
            <span>Choose Another Image</span>
          </button>
        )}
      </header>

      {/* Error banner */}
      {error && (
        <div className="alert-banner" role="alert">
          <AlertCircle size={16} aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      {/* Fallback notice (batch mode degraded to single-frame) */}
      {notice && (
        <div
          role="status"
          aria-live="polite"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '8px 12px',
            borderRadius: 'var(--radius-sm)',
            background: 'var(--surface-muted)',
            border: '1px solid var(--border)',
            color: 'var(--text-soft)',
            fontSize: '0.84rem',
            fontWeight: 500,
          }}
        >
          <span>{notice}</span>
        </div>
      )}

      {/* ── Idle: drop zone, or webcam burst panel when enabled ───────── */}
      {stage === 'idle' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <label
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              fontSize: '0.88rem',
              fontWeight: 600,
              color: 'var(--text-soft)',
              cursor: 'pointer',
              alignSelf: 'flex-start',
            }}
          >
            <input
              type="checkbox"
              checked={burstMode}
              disabled={burstBusy}
              onChange={handleBurstToggle}
            />
            Classroom burst (multi-frame)
          </label>

          {burstMode ? (
            <BurstCapturePanel
              videoRef={videoRef}
              isActive={cameraActive}
              error={cameraError}
              busy={burstBusy}
              onStartCamera={startCamera}
              onBurst={handleBurstSubmit}
            />
          ) : (
            <DropZone onFile={handleFile} />
          )}
        </div>
      )}

      {/* ── Preview: show image + file info + submit button ───────────── */}
      {stage === 'preview' && previewUrl && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* File info chip */}
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              padding: '6px 12px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--accent-soft)',
              border: '1px solid var(--accent-border)',
              color: 'var(--accent)',
              fontSize: '0.84rem',
              fontWeight: 600,
              alignSelf: 'flex-start',
            }}
          >
            <ImagePlus size={14} aria-hidden="true" />
            <span>{selectedFile.name}</span>
            <span style={{ opacity: 0.7 }}>({formatBytes(selectedFile.size)})</span>
          </div>

          {/* Image preview */}
          <div style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border)' }}>
            <img
              src={previewUrl}
              alt="Selected file preview"
              style={{ display: 'block', width: '100%', height: 'auto' }}
            />
          </div>

          {/* Action row */}
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button type="button" className="solid-btn" onClick={handleRecognize}>
              <ScanFace size={15} aria-hidden="true" />
              <span>Recognize</span>
            </button>
            <button type="button" className="ghost-btn" onClick={handleReset}>
              <span>Choose different image</span>
            </button>
          </div>
        </div>
      )}

      {/* ── Loading ───────────────────────────────────────────────────── */}
      {stage === 'loading' && (
        <div style={{ position: 'relative' }}>
          {previewUrl && (
            <div style={{ opacity: 0.4, borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border)' }}>
              <img
                src={previewUrl}
                alt=""
                aria-hidden="true"
                style={{ display: 'block', width: '100%', height: 'auto' }}
              />
            </div>
          )}
          <div
            style={{
              position: previewUrl ? 'absolute' : 'static',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Spinner
              label={
                burstMode
                  ? 'Running multi-frame batch pipeline...'
                  : 'Running recognition pipeline...'
              }
            />
          </div>
        </div>
      )}

      {/* ── Result ───────────────────────────────────────────────────── */}
      {stage === 'result' && result && previewUrl && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Stats row */}
          <StatsRow
            detectionCount={result.detection_count}
            matchCount={result.match_count}
          />

          {/* Annotated image */}
          <div style={{ border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
            <AnnotatedImage
              dataUrl={previewUrl}
              detections={result.detections}
              origWidth={result.image_width}
              origHeight={result.image_height}
            />
          </div>

          {/* Processed timestamp */}
          {result.processed_at && (
            <p className="meta-note" style={{ textAlign: 'right' }}>
              Processed at {new Date(result.processed_at).toLocaleString()}
            </p>
          )}

          {/* Results table */}
          <ResultsTable detections={result.detections} />

          {/* Reset button (bottom) */}
          <div style={{ display: 'flex', justifyContent: 'flex-start', paddingTop: 4 }}>
            <button type="button" className="ghost-btn" onClick={handleReset}>
              <RefreshCcw size={14} aria-hidden="true" />
              <span>Choose Another Image</span>
            </button>
          </div>
        </div>
      )}
    </section>
  )
}
