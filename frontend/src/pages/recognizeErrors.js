// ATT-053: extracted from Recognize.jsx so the mapping is unit-testable
// without tripping react-refresh's only-export-components rule.
// Maps an axios error onto the user-facing banner text. The
// `__REDIRECT_LOGIN__` sentinel means "session expired — send the user to
// /login" (callers navigate instead of rendering the message).
export function resolveError(err) {
  const status = err?.response?.status
  const detail = err?.response?.data?.detail
  if (status === 401) return '__REDIRECT_LOGIN__'
  if (status === 403) return 'Insufficient role: recognition requires instructor or admin access.'
  if (status === 413) return 'Capture too large. Please choose an image under the 10 MB limit.'
  if (status === 400) return detail || 'Image could not be decoded. Please upload a valid JPEG or PNG.'
  if (status === 422) return detail || 'Invalid request parameters.'
  if (status === 503) return 'Recognition service is temporarily unavailable. Please try again shortly.'
  if (status) return detail || `Request failed with HTTP status ${status}.`
  return err.message || 'A network error occurred. Check your connection and try again.'
}
