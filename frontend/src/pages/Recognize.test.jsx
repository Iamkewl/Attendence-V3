import { describe, expect, it } from 'vitest'
import { resolveError } from './recognizeErrors'

const httpError = (status, detail) =>
  Object.assign(new Error(`Request failed with status code ${status}`), {
    response: { status, data: detail === undefined ? undefined : { detail } },
  })

// ATT-053: the error banner mapping used by Recognize's photo and burst flows.
describe('Recognize resolveError', () => {
  it('maps 401 to the login-redirect sentinel', () => {
    expect(resolveError(httpError(401))).toBe('__REDIRECT_LOGIN__')
  })

  it('explains insufficient role on 403', () => {
    expect(resolveError(httpError(403))).toBe(
      'Insufficient role: recognition requires instructor or admin access.',
    )
  })

  it('explains the 10 MB limit on 413', () => {
    expect(resolveError(httpError(413))).toBe(
      'Capture too large. Please choose an image under the 10 MB limit.',
    )
  })

  it('prefers server-provided detail on 400/422', () => {
    expect(resolveError(httpError(400, 'Unsupported pixel format'))).toBe(
      'Unsupported pixel format',
    )
    expect(resolveError(httpError(422))).toBe('Invalid request parameters.')
  })

  it('falls back to friendly text without an HTTP response', () => {
    expect(resolveError(new Error('Network Error'))).toBe('Network Error')
    expect(resolveError({})).toBe(
      'A network error occurred. Check your connection and try again.',
    )
  })
})
