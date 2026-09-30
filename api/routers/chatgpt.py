"""
Sign in with ChatGPT endpoints (ChatGPT plan usage).

The OAuth redirect must be a 127.0.0.1 loopback URL, so the flow is:
POST /chatgpt/authorize → open the returned URL → the browser lands on
http://127.0.0.1:1455/auth/callback?... (which won't load) → the user pastes
that URL into POST /chatgpt/callback. See open_notebook/ai/chatgpt_plan.py.
"""

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.credentials_service import require_encryption_key
from open_notebook.ai import chatgpt_plan
from open_notebook.domain.credential import Credential
from open_notebook.exceptions import ConfigurationError, InvalidInputError

router = APIRouter(prefix="/chatgpt", tags=["chatgpt"])


class AuthorizeRequest(BaseModel):
    credential_id: Optional[str] = Field(
        None, description="Re-authorize this existing ChatGPT credential"
    )


class AuthorizeResponse(BaseModel):
    authorization_url: str
    redirect_uri: str


class CallbackRequest(BaseModel):
    callback_url: str = Field(
        ..., description="The full 127.0.0.1 URL the browser was redirected to"
    )


class AccountResponse(BaseModel):
    credential_id: Optional[str]
    email: Optional[str]
    client_id: Optional[str]
    plan_usage_enabled: bool
    expires_at: Optional[int]
    usage_url: str


@router.post("/authorize", response_model=AuthorizeResponse)
async def authorize(request: AuthorizeRequest):
    try:
        require_encryption_key()
    except ValueError as e:
        raise ConfigurationError(str(e)) from e
    result = await chatgpt_plan.start_authorization(request.credential_id)
    return AuthorizeResponse(
        authorization_url=result["authorization_url"],
        redirect_uri=result["redirect_uri"],
    )


@router.post("/callback", response_model=AccountResponse)
async def callback(request: CallbackRequest):
    try:
        cred = await chatgpt_plan.complete_authorization(request.callback_url)
    except chatgpt_plan.ChatGPTPlanError as e:
        raise InvalidInputError(str(e)) from e
    return AccountResponse(**chatgpt_plan.credential_summary(cred))


@router.get("/accounts", response_model=List[AccountResponse])
async def accounts():
    creds = await Credential.get_by_provider(chatgpt_plan.PROVIDER)
    return [AccountResponse(**chatgpt_plan.credential_summary(c)) for c in creds]
