import { afterEach, describe, expect, it, vi } from 'vitest'
import client from './client'

// ATT-053 smoke tests for the shared axios instance. Transport is faked via a
// custom adapter — no real network, backend, or module mocking needed here.
describe('api/client', () => {
  const originalAdapter = client.defaults.adapter

  afterEach(() => {
    client.defaults.adapter = originalAdapter
    vi.restoreAllMocks()
  })

  it('ships the shared instance defaults (JSON content type, cookies, timeout)', () => {
    expect(client.defaults.headers['Content-Type']).toBe('application/json')
    expect(client.defaults.withCredentials).toBe(true)
    expect(client.defaults.timeout).toBe(15000)
    expect(client.defaults.baseURL).toBe(import.meta.env.VITE_API_URL || '')
  })

  it('registers no request interceptors and exactly one response interceptor', () => {
    expect(client.interceptors.request.handlers).toHaveLength(0)
    expect(client.interceptors.response.handlers).toHaveLength(1)
  })

  it('replays a 401 request once after refreshing the session cookie', async () => {
    const refreshPost = vi.spyOn(client, 'post').mockResolvedValue({ status: 200 })
    const adapter = vi.fn(async (config) => ({
      data: { ok: true },
      status: 200,
      statusText: 'OK',
      headers: {},
      config,
    }))
    client.defaults.adapter = adapter

    const rejected = Object.assign(new Error('Request failed with status code 401'), {
      config: { url: '/api/v1/students', method: 'get' },
      response: { status: 401 },
    })

    const result = await client.interceptors.response.handlers[0].rejected(rejected)

    expect(refreshPost).toHaveBeenCalledTimes(1)
    expect(refreshPost.mock.calls[0][0]).toBe('/api/v1/auth/refresh')
    expect(adapter).toHaveBeenCalledTimes(1)
    expect(result.config._retry).toBe(true)
    expect(result.data).toEqual({ ok: true })
  })

  it('rejects non-401 errors without attempting a refresh', async () => {
    const refreshPost = vi.spyOn(client, 'post')
    const rejected = Object.assign(new Error('boom'), {
      config: { url: '/api/v1/students', method: 'get' },
      response: { status: 500 },
    })

    await expect(
      client.interceptors.response.handlers[0].rejected(rejected),
    ).rejects.toBe(rejected)
    expect(refreshPost).not.toHaveBeenCalled()
  })

  it('never triggers the refresh flow on auth lifecycle routes', async () => {
    const refreshPost = vi.spyOn(client, 'post')
    for (const url of [
      '/api/v1/auth/login',
      '/api/v1/auth/refresh',
      '/api/v1/auth/logout',
    ]) {
      const rejected = Object.assign(new Error('401 on auth route'), {
        config: { url, method: 'post' },
        response: { status: 401 },
      })
      await expect(
        client.interceptors.response.handlers[0].rejected(rejected),
      ).rejects.toBe(rejected)
    }
    expect(refreshPost).not.toHaveBeenCalled()
  })
})
