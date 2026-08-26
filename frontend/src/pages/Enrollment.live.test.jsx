import { act, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Enrollment from './Enrollment'
import { PREVIEW_INTERVAL_MS } from './enrollLogic'

// Live guided mode: the preview POST drives reason chips + pose guidance.
// The webcam/canvas seams are stubbed at jsdom boundaries (getUserMedia,
// toBlob, videoWidth) so no real camera or backend is involved.

const { postMock } = vi.hoisted(() => ({ postMock: vi.fn() }))
vi.mock('../api/client', () => ({
  default: {
    get: vi.fn(() => Promise.resolve({ data: [] })),
    post: postMock,
  },
}))

vi.mock('../auth/AuthContext', () => ({
  useAuth: () => ({ user: { id: 'user-1', role: 'admin' } }),
}))

vi.mock('./WebcamCapture', () => ({
  default: function WebcamCaptureStub() {
    return <div>webcam-manual-stub</div>
  },
}))

const PREVIEW_RESPONSE = {
  ok: false,
  detected: true,
  num_faces: 2,
  bbox: [0.25, 0.25, 0.5, 0.5],
  quality_score: 0.9,
  reasons: ['MULTIPLE_FACES'],
}

function stubBrowserCapturePrerequisites() {
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: vi.fn(() => Promise.resolve({ getTracks: () => [] })),
    },
  })
  HTMLCanvasElement.prototype.toBlob = function toBlobStub(callback) {
    callback(new Blob(['frame'], { type: 'image/jpeg' }))
  }
  // jsdom has no canvas backend; a minimal 2D context keeps the sampler and
  // overlay drawing functional under test.
  const ctxStub = { clearRect: () => {}, drawImage: () => {}, strokeRect: () => {} }
  HTMLCanvasElement.prototype.getContext = () => ctxStub
}

async function enterLiveModeWithCamera(container) {
  // StudentPicker's list fetch resolves on a microtask; flush before querying.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
  fireEvent.click(screen.getByRole('button', { name: /enter id manually/i }))
  const input = screen.getByLabelText('Student UUID')
  fireEvent.change(input, { target: { value: 'stu-live' } })
  fireEvent.click(screen.getByRole('button', { name: /select/i }))

  fireEvent.click(screen.getByRole('button', { name: /live guided mode/i }))
  fireEvent.click(screen.getByRole('button', { name: /enable live camera/i }))

  // getUserMedia resolves on a microtask; flush so the <video> mounts.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })

  const video = container.querySelector('video')
  Object.defineProperty(video, 'videoWidth', { configurable: true, value: 640 })
  Object.defineProperty(video, 'videoHeight', { configurable: true, value: 480 })
}

describe('Enrollment live guided mode', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    stubBrowserCapturePrerequisites()
    postMock.mockReset()
    postMock.mockResolvedValue({ status: 200, data: PREVIEW_RESPONSE })
  })

  it('samples the preview endpoint and renders reason chips from the response', async () => {
    const { container } = render(
      <MemoryRouter initialEntries={['/enroll']}>
        <Enrollment />
      </MemoryRouter>,
    )

    await enterLiveModeWithCamera(container)

    // One sampler tick: canvas JPEG -> POST /students/enroll/preview -> chips.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PREVIEW_INTERVAL_MS + 50)
    })

    expect(screen.getByText('MULTIPLE_FACES')).toBeTruthy()
    expect(postMock).toHaveBeenCalledTimes(1)
    const [url, body, config] = postMock.mock.calls[0]
    expect(url).toBe('/api/v1/students/enroll/preview')
    expect(body).toBeInstanceOf(FormData)
    expect(body.get('image_file')).toBeInstanceOf(Blob)
    // Same ATT-034 contract as manual enrollment: axios computes the boundary.
    expect(config.headers['Content-Type']).toBeUndefined()

    // Manual capture remains reachable from the live mode toggle.
    fireEvent.click(screen.getByRole('button', { name: /manual capture/i }))
    expect(screen.getByText('webcam-manual-stub')).toBeTruthy()

    vi.useRealTimers()
  })
})
