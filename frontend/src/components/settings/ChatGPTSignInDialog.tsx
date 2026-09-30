'use client'

import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { AlertCircle, ExternalLink, Loader2 } from 'lucide-react'
import { useTranslation } from '@/lib/hooks/use-translation'
import { CREDENTIAL_QUERY_KEYS } from '@/lib/hooks/use-credentials'
import { MODEL_QUERY_KEYS } from '@/lib/hooks/use-models'
import { chatgptApi, ChatGPTAccount, CHATGPT_USAGE_URL } from '@/lib/api/chatgpt'
import { getApiErrorMessage } from '@/lib/utils/error-handler'

export function ChatGPTLogo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className} fill="currentColor">
      <path d="M22.28 9.82a5.98 5.98 0 0 0-.52-4.91 6.05 6.05 0 0 0-6.51-2.9A6.07 6.07 0 0 0 4.98 4.18a5.99 5.99 0 0 0-4 2.9 6.05 6.05 0 0 0 .75 7.1 5.98 5.98 0 0 0 .51 4.91 6.05 6.05 0 0 0 6.52 2.9A5.98 5.98 0 0 0 13.26 24a6.06 6.06 0 0 0 5.77-4.21 5.99 5.99 0 0 0 4-2.9 6.06 6.06 0 0 0-.75-7.07zm-9.02 12.61a4.48 4.48 0 0 1-2.88-1.04l.14-.08 4.78-2.76a.8.8 0 0 0 .39-.68v-6.74l2.02 1.17a.07.07 0 0 1 .04.05v5.58a4.5 4.5 0 0 1-4.49 4.5zm-9.66-4.13a4.47 4.47 0 0 1-.53-3.01l.14.09 4.78 2.76a.77.77 0 0 0 .78 0l5.84-3.37v2.33a.08.08 0 0 1-.03.06l-4.84 2.79a4.5 4.5 0 0 1-6.14-1.65zM2.34 7.9a4.49 4.49 0 0 1 2.37-1.97v5.68a.77.77 0 0 0 .39.68l5.81 3.35-2.02 1.17a.08.08 0 0 1-.07 0l-4.83-2.79A4.5 4.5 0 0 1 2.34 7.87zm16.6 3.86-5.83-3.39 2.02-1.16a.08.08 0 0 1 .07 0l4.83 2.79a4.49 4.49 0 0 1-.68 8.1v-5.68a.79.79 0 0 0-.41-.66zm2.01-3.02-.14-.09-4.77-2.78a.78.78 0 0 0-.79 0L9.41 9.23V6.9a.07.07 0 0 1 .03-.06l4.83-2.79a4.5 4.5 0 0 1 6.68 4.66zM8.31 12.86 6.29 11.7a.08.08 0 0 1-.04-.06V6.07a4.5 4.5 0 0 1 7.38-3.45l-.14.08-4.78 2.76a.8.8 0 0 0-.39.68zm1.1-2.37 2.6-1.5 2.61 1.5v3l-2.6 1.5-2.61-1.5z" />
    </svg>
  )
}

interface ChatGPTSignInDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Re-authorize this existing ChatGPT credential instead of adding an account. */
  credentialId?: string
}

type Step = 'start' | 'paste' | 'done'

export function ChatGPTSignInDialog({ open, onOpenChange, credentialId }: ChatGPTSignInDialogProps) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [step, setStep] = useState<Step>('start')
  const [callbackUrl, setCallbackUrl] = useState('')
  const [redirectUri, setRedirectUri] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [account, setAccount] = useState<ChatGPTAccount | null>(null)

  const start = async () => {
    setError(null)
    setPending(true)
    // Open the tab synchronously so popup blockers allow it, then navigate it.
    const tab = window.open('about:blank', '_blank')
    try {
      const { authorization_url, redirect_uri } = await chatgptApi.authorize(credentialId)
      setRedirectUri(redirect_uri)
      if (tab) {
        tab.opener = null
        tab.location.href = authorization_url
      } else {
        window.location.assign(authorization_url)
      }
      setStep('paste')
    } catch (e) {
      tab?.close()
      setError(getApiErrorMessage(e, t))
    } finally {
      setPending(false)
    }
  }

  const finish = async () => {
    setError(null)
    setPending(true)
    try {
      const result = await chatgptApi.callback(callbackUrl)
      setAccount(result)
      setStep('done')
      queryClient.invalidateQueries({ queryKey: CREDENTIAL_QUERY_KEYS.all })
      queryClient.invalidateQueries({ queryKey: MODEL_QUERY_KEYS.providers })
    } catch (e) {
      setError(getApiErrorMessage(e, t))
    } finally {
      setPending(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ChatGPTLogo className="h-5 w-5" />
            {step === 'done' ? t('chatgptPlan.usingPlanTitle') : t('chatgptPlan.title')}
          </DialogTitle>
          <DialogDescription>
            {step === 'done' ? t('chatgptPlan.usingPlanDescription') : t('chatgptPlan.description')}
          </DialogDescription>
        </DialogHeader>

        {step === 'start' && (
          <div className="space-y-3 text-sm text-muted-foreground">
            <p>{t('chatgptPlan.eligibility')}</p>
            <p>{t('chatgptPlan.selfHostedNote')}</p>
          </div>
        )}

        {step === 'paste' && (
          <div className="space-y-3">
            <ol className="list-decimal pl-5 space-y-1 text-sm text-muted-foreground">
              <li>{t('chatgptPlan.pasteStep1')}</li>
              <li>
                {t('chatgptPlan.pasteStep2')}{' '}
                <code className="text-xs bg-muted px-1 py-0.5 rounded break-all">{redirectUri}…</code>
              </li>
              <li>{t('chatgptPlan.pasteStep3')}</li>
            </ol>
            <div className="space-y-1.5">
              <Label htmlFor="chatgpt-callback">{t('chatgptPlan.callbackLabel')}</Label>
              <Input
                id="chatgpt-callback"
                placeholder="http://127.0.0.1:1455/auth/callback?code=…&state=…"
                value={callbackUrl}
                onChange={e => setCallbackUrl(e.target.value)}
                autoComplete="off"
                spellCheck={false}
              />
            </div>
          </div>
        )}

        {step === 'done' && account && (
          <div className="space-y-3 text-sm">
            <p>
              {t('chatgptPlan.signedInAs')} <strong>{account.email ?? account.client_id}</strong>
            </p>
            {!account.plan_usage_enabled && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{t('chatgptPlan.planNotGranted')}</AlertDescription>
              </Alert>
            )}
            <p className="text-muted-foreground">{t('chatgptPlan.nextStep')}</p>
          </div>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <DialogFooter className="gap-2 sm:gap-0">
          {step === 'start' && (
            <Button onClick={start} disabled={pending} className="gap-2 bg-black text-white hover:bg-black/85">
              {pending ? <Loader2 className="h-4 w-4 animate-spin" /> : <ChatGPTLogo className="h-4 w-4" />}
              {t('chatgptPlan.continueWithChatGPT')}
            </Button>
          )}
          {step === 'paste' && (
            <>
              <Button variant="ghost" onClick={start} disabled={pending}>
                {t('chatgptPlan.openAgain')}
              </Button>
              <Button onClick={finish} disabled={pending || !callbackUrl.trim()}>
                {pending && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
                {t('chatgptPlan.finish')}
              </Button>
            </>
          )}
          {step === 'done' && (
            <>
              <Button variant="ghost" asChild>
                <a href={CHATGPT_USAGE_URL} target="_blank" rel="noopener noreferrer" className="gap-1">
                  {t('chatgptPlan.manageUsage')}
                  <ExternalLink className="h-3 w-3" />
                </a>
              </Button>
              <Button onClick={() => onOpenChange(false)}>{t('chatgptPlan.gotIt')}</Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
