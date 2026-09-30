'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Plus, Check, X, ExternalLink } from 'lucide-react'
import { useTranslation } from '@/lib/hooks/use-translation'
import { Credential } from '@/lib/api/credentials'
import { ProviderInfo } from '@/lib/api/providers'
import { Model, ModelDefaults } from '@/lib/types/models'
import {
  getTypeIcon,
  getTypeColor,
  getTypeLabel,
  TYPE_COLOR_INACTIVE,
} from '@/lib/providers'
import { CredentialFormDialog } from './CredentialFormDialog'
import { CredentialItem } from './CredentialItem'
import { ChatGPTSignInDialog, ChatGPTLogo } from './ChatGPTSignInDialog'
import { CHATGPT_USAGE_URL } from '@/lib/api/chatgpt'

interface ProviderSectionProps {
  provider: ProviderInfo
  credentials: Credential[]
  models: Model[]
  defaults: ModelDefaults | null
  allCredentials: Credential[]
  encryptionReady: boolean
}

export function ProviderSection({
  provider,
  credentials,
  models,
  defaults,
  allCredentials,
  encryptionReady,
}: ProviderSectionProps) {
  const { t } = useTranslation()
  const [addOpen, setAddOpen] = useState(false)

  const displayName = provider.display_name || provider.name
  const modalities = provider.modalities.length > 0 ? provider.modalities : ['language']
  const hasCredentials = credentials.length > 0
  const isChatGPT = provider.name === 'chatgpt'

  // Models linked to any credential of this provider
  const providerModels = models.filter(m =>
    credentials.some(c => c.id === m.credential)
  )
  const activeTypes = new Set<string>(providerModels.map(m => m.type))

  return (
    <Card className={hasCredentials ? 'border-l-2 border-l-fern' : undefined}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 flex-wrap">
            <CardTitle className={`text-lg capitalize ${hasCredentials ? '' : 'text-muted-foreground'}`}>{displayName}</CardTitle>
            <div className="flex items-center gap-1">
              {modalities.map((type) => (
                <Badge
                  key={type}
                  variant="secondary"
                  className={`text-xs gap-1 ${activeTypes.has(type) ? getTypeColor(type) : TYPE_COLOR_INACTIVE}`}
                >
                  {getTypeIcon(type)}
                  <span className="hidden sm:inline">{getTypeLabel(type)}</span>
                </Badge>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {hasCredentials ? (
              <span className="inline-flex items-center gap-1.5 text-xs font-medium text-fern">
                <Check className="h-3 w-3" />
                {t('apiKeys.configured')}
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <X className="h-3 w-3" />
                {t('apiKeys.notConfigured')}
              </span>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        {credentials.map(cred => (
          <CredentialItem
            key={cred.id}
            credential={cred}
            models={models}
            defaults={defaults}
            allCredentials={allCredentials}
          />
        ))}

        {isChatGPT ? (
          <>
            <p className="text-sm text-muted-foreground">{t('chatgptPlan.cardDescription')}</p>
            <Button
              size="sm"
              onClick={() => setAddOpen(true)}
              className="w-full gap-2 bg-black text-white hover:bg-black/85"
              disabled={!encryptionReady}
            >
              <ChatGPTLogo className="h-4 w-4" />
              {t('chatgptPlan.continueWithChatGPT')}
            </Button>
            {hasCredentials && (
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="inline-flex items-center gap-1.5">
                  <ChatGPTLogo className="h-3 w-3" />
                  {t('chatgptPlan.usingPlan')}
                </span>
                <a
                  href={CHATGPT_USAGE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-primary hover:underline"
                >
                  {t('chatgptPlan.manageUsage')}
                  <ExternalLink className="h-3 w-3" />
                </a>
              </div>
            )}
          </>
        ) : (
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAddOpen(true)}
            className="w-full gap-2"
            disabled={!encryptionReady}
          >
            <Plus className="h-4 w-4" />
            {t('apiKeys.addConfig')}
          </Button>
        )}
      </CardContent>

      {addOpen && isChatGPT && (
        <ChatGPTSignInDialog open={addOpen} onOpenChange={setAddOpen} />
      )}

      {addOpen && !isChatGPT && (
        <CredentialFormDialog
          open={addOpen}
          onOpenChange={setAddOpen}
          provider={provider.name}
        />
      )}
    </Card>
  )
}
