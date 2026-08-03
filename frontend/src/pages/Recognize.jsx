import { useCallback, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, ImagePlus, RefreshCcw, ScanFace, UploadCloud } from 'lucide-react'
import client from '../api/client'
import AnnotatedImage from '../components/AnnotatedImage'

const RECOGNIZE_ENDPOINT = '/api/v1/inference/photo'
const MAX_FILE_BYTES = 10 * 1024 * 1024 // 10 MB

function formatBytes(n) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

function resolveError(err) {
  const status = err?.response?.status
  const detail = err?.response?.data?.detail
  if (status === 400) return detail || 'Image could not be decoded. Please upload a valid JPEG or PNG.'
  if (status === 401) return '__REDIRECT_LOGIN__'
  if (status === 403) return 'You do not have permission to use this feature.'
  if (status === 413) return 'Image exceeds the 10 MB limit. Please choose a smaller file.'
  if (status === 422) return detail || 'Invalid request parameters.'
  if (status === 503) return 'Recognition service is temporarily unavailable. Please try again shortly.'
  return detail || err.message || 'An unexpected error occurred.'
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
                  {det.match ? det.match.student_full_name : '—'}
                </td>
                <td style={{ fontFamily: 'monospace', fontSize: '0.82rem' }}>
                  {det.match ? det.match.student_number : '—'}
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

// ── Main Page ──────────────────────────────────────────────────────────────
export default function Recognize() {
  const navigate = useNavigate()

  // 'idle' | 'preview' | 'loading' | 'result'
  const [stage, setStage] = useState('idle')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

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

  const handleReset = useCallback(() => {
    setStage('idle')
    setSelectedFile(null)
    setPreviewUrl(null)
    setError('')
    setResult(null)
  }, [])

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

      {/* ── Idle: show drop zone ──────────────────────────────────────── */}
      {stage === 'idle' && (
        <DropZone onFile={handleFile} />
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
              alt="Selected photo preview"
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
            <Spinner label="Running recognition pipeline..." />
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
