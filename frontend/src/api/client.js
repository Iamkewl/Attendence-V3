import axios from 'axios'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  withCredentials: true,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
})

let refreshPromise = null

const isAuthLifecycleRoute = (url = '') =>
  url.includes('/api/v1/auth/login') ||
  url.includes('/api/v1/auth/refresh') ||
  url.includes('/api/v1/auth/logout')

const ensureRefreshedSession = async () => {
  if (!refreshPromise) {
    refreshPromise = client
      .post('/api/v1/auth/refresh', null, { skipAuthRefresh: true })
      .finally(() => {
        refreshPromise = null
      })
  }

  return refreshPromise
}

client.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error?.config
    const statusCode = error?.response?.status

    if (!originalRequest || originalRequest.skipAuthRefresh || statusCode !== 401) {
      return Promise.reject(error)
    }

    const requestUrl = String(originalRequest.url ?? '')
    if (originalRequest._retry || isAuthLifecycleRoute(requestUrl)) {
      return Promise.reject(error)
    }

    originalRequest._retry = true

    try {
      await ensureRefreshedSession()
      return client(originalRequest)
    } catch (refreshError) {
      return Promise.reject(refreshError)
    }
  },
)

export default client