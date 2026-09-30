import apiClient from './client'

// Sign in with ChatGPT (ChatGPT plan usage). See api/routers/chatgpt.py.
export interface ChatGPTAccount {
  credential_id: string | null
  email: string | null
  client_id: string | null
  plan_usage_enabled: boolean
  expires_at: number | null
  usage_url: string
}

export interface ChatGPTAuthorizeResponse {
  authorization_url: string
  redirect_uri: string
}

export const CHATGPT_USAGE_URL = 'https://chatgpt.com/settings/usage'

export const chatgptApi = {
  /** Start OAuth. Pass a credential id to re-authorize an existing account. */
  authorize: async (credentialId?: string): Promise<ChatGPTAuthorizeResponse> => {
    const response = await apiClient.post<ChatGPTAuthorizeResponse>('/chatgpt/authorize', {
      credential_id: credentialId ?? null,
    })
    return response.data
  },

  /** Finish OAuth with the 127.0.0.1 URL the browser was redirected to. */
  callback: async (callbackUrl: string): Promise<ChatGPTAccount> => {
    const response = await apiClient.post<ChatGPTAccount>('/chatgpt/callback', {
      callback_url: callbackUrl,
    })
    return response.data
  },
}
