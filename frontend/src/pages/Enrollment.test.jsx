import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Enrollment from './Enrollment'

// ATT-053: verify the enrollment POST goes out as FormData with NO explicit
// Content-Type header (ATT-034 contract — axios must compute the boundary).
// The shared client module is mocked wholesale; the webcam seam is stubbed so
// no real camera or backend is required.

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
  default: function WebcamCaptureStub({ onCapture, onSubmit }) {
    return (
      <div>
        <button type="button" onClick={() => onCapture(new Blob(['frame-bytes'], { type: 'image/jpeg' }))}>
          capture-stub
        </button>
        <button type="button" onClick={() => onSubmit()}>
          submit-stub
        </button>
      </div>
    )
  },
}))

function renderEnrollment() {
  return render(
    <MemoryRouter initialEntries={['/enroll']}>
      <Enrollment />
    </MemoryRouter>,
  )
}

async function selectStudentManually() {
  fireEvent.click(await screen.findByRole('button', { name: /enter id manually/i }))
  const input = await screen.findByLabelText('Student UUID')
  fireEvent.change(input, { target: { value: 'stu-123' } })
  fireEvent.click(screen.getByRole('button', { name: /select/i }))
}

describe('Enrollment submit flow', () => {
  beforeEach(() => {
    postMock.mockReset()
    postMock.mockResolvedValue({
      status: 201,
      data: {
        id: 'tpl-1',
        student_id: 'stu-123',
        quality_score: 0.92,
        pose_label: 'frontal',
        created_at: '2026-08-25T00:00:00Z',
      },
    })
  })

  it('posts multipart FormData without an explicit Content-Type header', async () => {
    renderEnrollment()
    await selectStudentManually()

    fireEvent.click(await screen.findByRole('button', { name: /capture-stub/i }))
    fireEvent.click(screen.getByRole('button', { name: /submit-stub/i }))

    await waitFor(() => screen.getByText('Enrollment Successful'))

    expect(postMock).toHaveBeenCalledTimes(1)
    const [url, body, config] = postMock.mock.calls[0]
    expect(url).toBe('/api/v1/students/stu-123/enroll')
    expect(body).toBeInstanceOf(FormData)
    expect(body.get('image_file')).toBeInstanceOf(Blob)
    // The whole point of ATT-034: never hand-roll multipart Content-Type.
    expect(config.headers['Content-Type']).toBeUndefined()
  })

  it('surfaces a friendly message when enrollment fails with 413', async () => {
    postMock.mockRejectedValue(
      Object.assign(new Error('Request failed with status code 413'), {
        response: { status: 413, data: undefined },
      }),
    )

    renderEnrollment()
    await selectStudentManually()

    fireEvent.click(await screen.findByRole('button', { name: /capture-stub/i }))
    fireEvent.click(screen.getByRole('button', { name: /submit-stub/i }))

    const banner = await screen.findByRole('alert')
    expect(banner.textContent).toContain(
      'Capture too large. Please retake with a smaller image (max 10 MB).',
    )
  })
})
