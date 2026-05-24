import { useEffect, useRef, useCallback } from 'react'

const COLOR_MATCHED_LIVE = '#22c55e'    // green
const COLOR_UNKNOWN_LIVE = '#eab308'    // yellow
const COLOR_LIVENESS_FAIL = '#ef4444'  // red

function classifyDetection(det) {
  if (!det.is_live) return 'liveness_fail'
  if (det.match) return 'matched_live'
  return 'unknown_live'
}

function buildLabel(det) {
  const kind = classifyDetection(det)
  if (kind === 'liveness_fail') return 'Liveness failed'
  if (kind === 'matched_live') {
    const sim = det.match.cosine_similarity != null
      ? ` ${Math.round(det.match.cosine_similarity * 100)}%`
      : ''
    return `${det.match.student_full_name} (${det.match.student_number})${sim}`
  }
  // unknown live
  const sim = det.match?.cosine_similarity != null
    ? ` best ${Math.round(det.match.cosine_similarity * 100)}%`
    : ''
  return `Unknown${sim}`
}

function getColor(det) {
  const kind = classifyDetection(det)
  if (kind === 'matched_live') return COLOR_MATCHED_LIVE
  if (kind === 'unknown_live') return COLOR_UNKNOWN_LIVE
  return COLOR_LIVENESS_FAIL
}

export default function AnnotatedImage({ dataUrl, detections, origWidth, origHeight }) {
  const imgRef = useRef(null)
  const canvasRef = useRef(null)

  const draw = useCallback(() => {
    const img = imgRef.current
    const canvas = canvasRef.current
    if (!img || !canvas || !origWidth || !origHeight) return

    const renderedW = img.clientWidth
    const renderedH = img.clientHeight
    if (!renderedW || !renderedH) return

    canvas.width = renderedW
    canvas.height = renderedH

    const scaleX = renderedW / origWidth
    const scaleY = renderedH / origHeight

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, renderedW, renderedH)

    for (const det of detections) {
      const [bx, by, bw, bh] = det.bbox
      const x = bx * scaleX
      const y = by * scaleY
      const w = bw * scaleX
      const h = bh * scaleY
      const color = getColor(det)
      const label = buildLabel(det)

      // Box
      ctx.strokeStyle = color
      ctx.lineWidth = 2.5
      ctx.strokeRect(x, y, w, h)

      // Label chip background
      ctx.font = '600 11px Inter, ui-sans-serif, system-ui, sans-serif'
      const textMetrics = ctx.measureText(label)
      const chipPadH = 5
      const chipPadV = 3
      const chipW = textMetrics.width + chipPadH * 2
      const chipH = 18
      const chipX = x
      const chipY = y - chipH - 2 < 0 ? y + 2 : y - chipH - 2

      ctx.fillStyle = color
      ctx.beginPath()
      ctx.roundRect(chipX, chipY, chipW, chipH, 3)
      ctx.fill()

      // Label text
      ctx.fillStyle = '#fff'
      ctx.fillText(label, chipX + chipPadH, chipY + chipH - chipPadV - 1)
    }
  }, [detections, origWidth, origHeight])

  // Draw after image load and on resize
  useEffect(() => {
    const img = imgRef.current
    if (!img) return

    const handleLoad = () => draw()
    img.addEventListener('load', handleLoad)
    if (img.complete) draw()

    const ro = new ResizeObserver(() => draw())
    ro.observe(img)

    return () => {
      img.removeEventListener('load', handleLoad)
      ro.disconnect()
    }
  }, [draw])

  return (
    <div style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
      <img
        ref={imgRef}
        src={dataUrl}
        alt="Recognition result"
        style={{ display: 'block', width: '100%', height: 'auto', borderRadius: 'var(--radius-md)' }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          pointerEvents: 'none',
          width: '100%',
          height: '100%',
          borderRadius: 'var(--radius-md)',
        }}
        aria-hidden="true"
      />
    </div>
  )
}
