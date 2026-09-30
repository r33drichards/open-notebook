"""
Sign in with ChatGPT — ChatGPT plan usage for Open Notebook.

Implements OpenAI's open-source "ChatGPT plan usage" flow
(https://developers.openai.com/siwc/token-sharing-open-source):

1. The instance persists a stable ``ext_agent_host_id`` (``urn:uuid:...``).
2. The user starts OAuth (Authorization Code + PKCE + OIDC). First-time
   registration uses ``client_id=dynamic_agent_client``; the callback returns
   the issued ``client_id`` that is reused for later sign-ins.
3. OpenAI only accepts ``http://127.0.0.1:<port>/auth/callback`` redirects. A
   self-hosted instance is usually not on the user's machine, so the browser
   lands on a loopback URL that doesn't load and the user pastes that URL back
   into Open Notebook (the "self-hosted VM" pattern from the docs: OAuth runs
   in the user's local browser, the instance owns the credentials).
4. The code is exchanged, the ID token is verified against OpenAI's JWKS and
   the ``chatgpt.tokens.use.direct`` scope is checked.
5. Tokens are stored encrypted in a ``credential`` record (provider
   ``chatgpt``) and refreshed near expiry.
6. Inference goes to ``POST https://api.openai.com/v1/responses`` with the
   access token, ``store: false`` and ``stream: true``.
"""

import asyncio
import base64
import hashlib
import json
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from esperanto import LanguageModel
from esperanto.common_types import ChatCompletion
from esperanto.common_types.response import Choice, Message
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from loguru import logger
from pydantic import SecretStr

from open_notebook.exceptions import ConfigurationError

PROVIDER = "chatgpt"
APP_NAME = "Open Notebook"

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
OPENID_CONFIG_URL = f"{ISSUER}/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
RESPONSES_URL = f"{RESOURCE}/responses"
MODELS_URL = f"{RESOURCE}/models"
REDIRECT_URI = "http://127.0.0.1:1455/auth/callback"
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {PLAN_SCOPE}"
USAGE_URL = "https://chatgpt.com/settings/usage"

HOST_RECORD_ID = "open_notebook:chatgpt_plan_host"
PENDING_TTL_SECONDS = 15 * 60
# Refresh when the access token has less than this many seconds left.
REFRESH_MARGIN_SECONDS = 5 * 60

# Errors that mean the refresh token is unusable and the user must sign in again.
TERMINAL_REFRESH_ERRORS = {
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
}


class ChatGPTPlanError(ConfigurationError):
    """Raised for ChatGPT plan sign-in, refresh and inference failures."""


# =============================================================================
# PKCE / pending authorization state
# =============================================================================


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _pkce_pair() -> Tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(48))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


@dataclass
class PendingAuthorization:
    state: str
    nonce: str
    code_verifier: str
    client_id: str  # DYNAMIC_CLIENT_ID for a new registration
    host_id: str
    credential_id: Optional[str]  # set when re-authorizing an existing account
    created_at: float = field(default_factory=time.time)

    @property
    def is_new_registration(self) -> bool:
        return self.client_id == DYNAMIC_CLIENT_ID


# The API runs as a single process, and the start/complete requests of one
# sign-in attempt both hit it, so in-memory state is sufficient here.
_pending: Dict[str, PendingAuthorization] = {}


def _prune_pending() -> None:
    cutoff = time.time() - PENDING_TTL_SECONDS
    for key in [k for k, v in _pending.items() if v.created_at < cutoff]:
        _pending.pop(key, None)


# =============================================================================
# Host id
# =============================================================================


async def get_host_id() -> str:
    """Return this instance's stable ext_agent_host_id, creating it once."""
    from open_notebook.database.repository import repo_query, repo_upsert

    rows = await repo_query(
        "SELECT * FROM ONLY $id", {"id": _record_id(HOST_RECORD_ID)}
    )
    row = rows[0] if isinstance(rows, list) and rows else rows
    if isinstance(row, dict) and row.get("host_id"):
        return row["host_id"]

    host_id = f"urn:uuid:{uuid.uuid4()}"
    await repo_upsert("open_notebook", HOST_RECORD_ID, {"host_id": host_id})
    return host_id


def _record_id(value: str):
    from open_notebook.database.repository import ensure_record_id

    return ensure_record_id(value)


# =============================================================================
# Credential storage helpers
# =============================================================================


def _meta(cred) -> Dict[str, Any]:
    return dict((cred.config or {}).get("chatgpt") or {})


def _tokens(cred) -> Dict[str, Any]:
    if not cred.api_key:
        return {}
    try:
        return json.loads(cred.api_key.get_secret_value())
    except (ValueError, TypeError):
        return {}


def _store(cred, tokens: Dict[str, Any], meta: Dict[str, Any]) -> None:
    cred.api_key = SecretStr(json.dumps(tokens))
    config = dict(cred.config or {})
    config["chatgpt"] = meta
    cred.config = config


def _token_record(token_response: Dict[str, Any], previous: Dict[str, Any]) -> Tuple[
    Dict[str, Any], Dict[str, Any]
]:
    """Split a token-endpoint response into secret tokens and plain metadata."""
    now = int(time.time())
    tokens = {
        "access_token": token_response["access_token"],
        "refresh_token": token_response.get("refresh_token")
        or previous.get("refresh_token"),
        "id_token": token_response.get("id_token") or previous.get("id_token"),
    }
    meta = {
        "expires_at": now + int(token_response.get("expires_in") or 3600),
        "scopes": sorted(str(token_response.get("scope") or "").split()),
        "saved_at": now,
    }
    return tokens, meta


def credential_summary(cred) -> Dict[str, Any]:
    """Non-secret account info for the UI."""
    meta = _meta(cred)
    return {
        "credential_id": str(cred.id) if cred.id else None,
        "email": meta.get("email"),
        "client_id": meta.get("client_id"),
        "plan_usage_enabled": PLAN_SCOPE in (meta.get("scopes") or []),
        "expires_at": meta.get("expires_at"),
        "usage_url": USAGE_URL,
    }


# =============================================================================
# Authorization
# =============================================================================


async def start_authorization(credential_id: Optional[str] = None) -> Dict[str, str]:
    """
    Build the OpenAI authorization URL.

    Without ``credential_id`` this registers a new client for a new ChatGPT
    account. With it, the account's issued client id is reused (reauthorization,
    e.g. after the refresh token expired or to enable plan usage).
    """
    from open_notebook.domain.credential import Credential

    _prune_pending()
    host_id = await get_host_id()
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)

    params: Dict[str, str] = {
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "ext_agent_host_id": host_id,
    }

    client_id = DYNAMIC_CLIENT_ID
    if credential_id:
        cred = await Credential.get(credential_id)
        if cred.provider != PROVIDER:
            raise ChatGPTPlanError("Credential is not a ChatGPT sign-in")
        meta = _meta(cred)
        client_id = meta.get("client_id") or DYNAMIC_CLIENT_ID
        id_token = _tokens(cred).get("id_token")
        if id_token:
            params["id_token_hint"] = id_token
        if meta.get("email"):
            params["login_hint"] = meta["email"]
        if PLAN_SCOPE not in (meta.get("scopes") or []):
            # The user is explicitly asking to enable plan usage after declining.
            params["prompt"] = "consent"

    if client_id == DYNAMIC_CLIENT_ID:
        params["agent_name_hint"] = APP_NAME
    params["client_id"] = client_id

    _pending[state] = PendingAuthorization(
        state=state,
        nonce=nonce,
        code_verifier=verifier,
        client_id=client_id,
        host_id=host_id,
        credential_id=credential_id,
    )
    return {
        "authorization_url": f"{AUTHORIZE_URL}?{urlencode(params)}",
        "redirect_uri": REDIRECT_URI,
        "state": state,
    }


def parse_callback(callback_url: str) -> Dict[str, str]:
    """Accept the full loopback URL (or just its query string) the user pasted."""
    value = (callback_url or "").strip()
    if not value:
        raise ChatGPTPlanError("Paste the URL your browser was redirected to")
    query = urlparse(value).query if "://" in value else value.lstrip("?")
    params = {k: v[0] for k, v in parse_qs(query).items() if v}
    if not params.get("state"):
        raise ChatGPTPlanError(
            "That URL has no 'state' parameter — copy the full address from the browser"
        )
    return params


async def _post_token(data: Dict[str, str]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            TOKEN_URL,
            data=data,
            headers={"Accept": "application/json"},
        )
    try:
        body = response.json()
    except ValueError:
        body = {"error": response.text[:200]}
    if response.status_code != 200:
        code = body.get("error") if isinstance(body, dict) else None
        if isinstance(code, dict):
            code = code.get("code") or code.get("message")
        raise ChatGPTPlanTokenError(str(code or response.status_code), body)
    return body


class ChatGPTPlanTokenError(ChatGPTPlanError):
    def __init__(self, code: str, body: Any):
        super().__init__(f"OpenAI token endpoint error: {code}")
        self.code = code
        self.body = body


def _verify_id_token(id_token: str, client_id: str, nonce: Optional[str]) -> Dict[str, Any]:
    import jwt

    signing_key = _jwks_client().get_signing_key_from_jwt(id_token)
    claims = jwt.decode(
        id_token,
        signing_key.key,
        algorithms=["RS256", "ES256"],
        audience=client_id,
        issuer=ISSUER,
        leeway=60,
        options={"require": ["exp", "iat", "sub", "iss", "aud"]},
    )
    if nonce is not None and claims.get("nonce") != nonce:
        raise ChatGPTPlanError("ID token nonce mismatch")
    return claims


_jwks = None


def _jwks_client():
    global _jwks
    if _jwks is None:
        import jwt

        _jwks = jwt.PyJWKClient(JWKS_URL, cache_keys=True, lifespan=3600)
    return _jwks


async def complete_authorization(callback_url: str):
    """Exchange the pasted callback for tokens and save the credential."""
    from open_notebook.domain.credential import Credential

    params = parse_callback(callback_url)
    pending = _pending.pop(params["state"], None)
    if not pending or pending.created_at < time.time() - PENDING_TTL_SECONDS:
        raise ChatGPTPlanError(
            "This sign-in attempt expired or was already used — start again"
        )

    if params.get("error"):
        if params["error"] == "access_denied":
            raise ChatGPTPlanError("Sign-in was cancelled in the browser")
        raise ChatGPTPlanError(
            f"OpenAI returned an error: {params.get('error_description') or params['error']}"
        )

    code = params.get("code")
    if not code:
        raise ChatGPTPlanError("The callback URL has no authorization code")

    returned_client_id = params.get("client_id")
    if pending.is_new_registration:
        if not returned_client_id or returned_client_id == DYNAMIC_CLIENT_ID:
            raise ChatGPTPlanError(
                "Registration incomplete: the callback did not include an issued client_id"
            )
        client_id = returned_client_id
    else:
        client_id = pending.client_id
        if returned_client_id and returned_client_id != client_id:
            raise ChatGPTPlanError("The callback belongs to a different registration")

    token_response = await _post_token(
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": pending.code_verifier,
            "redirect_uri": REDIRECT_URI,
            "resource": RESOURCE,
        }
    )
    id_token = token_response.get("id_token")
    if not id_token:
        raise ChatGPTPlanError("OpenAI did not return an ID token")
    claims = await asyncio.to_thread(
        _verify_id_token, id_token, client_id, pending.nonce
    )

    tokens, token_meta = _token_record(token_response, {})
    meta = {
        **token_meta,
        "client_id": client_id,
        "subject": claims["sub"],
        "issuer": claims["iss"],
        "email": claims.get("email"),
        "ext_agent_host_id": pending.host_id,
    }

    cred = None
    if pending.credential_id:
        cred = await Credential.get(pending.credential_id)
        previous = _meta(cred)
        if previous.get("subject") and previous["subject"] != claims["sub"]:
            raise ChatGPTPlanError(
                "You signed in as a different ChatGPT account than the one being renewed"
            )
    else:
        for existing in await Credential.get_by_provider(PROVIDER):
            if _meta(existing).get("client_id") == client_id:
                cred = existing
                break

    if cred is None:
        label = claims.get("email") or claims["sub"][:12]
        cred = Credential(
            name=f"ChatGPT ({label})",
            provider=PROVIDER,
            modalities=["language"],
        )
    _store(cred, tokens, meta)
    await cred.save()
    logger.info(
        f"ChatGPT sign-in saved for credential {cred.id} "
        f"(plan usage {'enabled' if PLAN_SCOPE in meta['scopes'] else 'NOT granted'})"
    )
    return cred


# =============================================================================
# Token refresh / revocation
# =============================================================================

_refresh_locks: Dict[str, asyncio.Lock] = {}


async def get_access_token(credential_id: str) -> str:
    """Return a valid access token for the credential, refreshing if needed."""
    from open_notebook.domain.credential import Credential

    lock = _refresh_locks.setdefault(str(credential_id), asyncio.Lock())
    async with lock:
        # Re-read inside the lock: another request (or the worker process) may
        # already have rotated the refresh token.
        cred = await Credential.get(credential_id)
        tokens, meta = _tokens(cred), _meta(cred)
        if not tokens.get("access_token"):
            raise ChatGPTPlanError("ChatGPT account is signed out — sign in again")
        if PLAN_SCOPE not in (meta.get("scopes") or []):
            raise ChatGPTPlanError(
                "ChatGPT plan usage was not authorized for this account — "
                "use 'Enable ChatGPT plan' in Settings → Models"
            )
        if int(meta.get("expires_at") or 0) - REFRESH_MARGIN_SECONDS > time.time():
            return tokens["access_token"]
        if not tokens.get("refresh_token"):
            raise ChatGPTPlanError("ChatGPT session expired — sign in again")

        try:
            response = await _post_token(
                {
                    "grant_type": "refresh_token",
                    "client_id": meta["client_id"],
                    "refresh_token": tokens["refresh_token"],
                    "resource": RESOURCE,
                }
            )
        except ChatGPTPlanTokenError as e:
            if e.code in TERMINAL_REFRESH_ERRORS:
                _store(cred, {"id_token": tokens.get("id_token")}, meta)
                await cred.save()
                raise ChatGPTPlanError(
                    "ChatGPT session ended — sign in again in Settings → Models"
                ) from e
            raise

        new_tokens, token_meta = _token_record(response, tokens)
        meta.update(token_meta)
        _store(cred, new_tokens, meta)
        await cred.save()
        return new_tokens["access_token"]


async def revoke(cred) -> bool:
    """Best-effort revocation of the renewable session. Returns True if confirmed."""
    tokens, meta = _tokens(cred), _meta(cred)
    if not tokens.get("refresh_token") or not meta.get("client_id"):
        return False
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            discovery = (await client.get(OPENID_CONFIG_URL)).json()
            endpoint = discovery.get("revocation_endpoint")
            if not endpoint:
                return False
            response = await client.post(
                endpoint,
                data={
                    "token": tokens["refresh_token"],
                    "token_type_hint": "refresh_token",
                    "client_id": meta["client_id"],
                },
            )
            return response.status_code == 200
    except Exception as e:
        logger.warning(f"ChatGPT token revocation failed: {e}")
        return False


# =============================================================================
# Models / inference
# =============================================================================


async def list_models(access_token: str) -> List[Dict[str, str]]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            MODELS_URL, headers={"Authorization": f"Bearer {access_token}"}
        )
    if response.status_code != 200:
        raise ChatGPTPlanError(
            f"Listing ChatGPT plan models failed ({response.status_code}): {response.text[:200]}"
        )
    data = response.json()
    models = data.get("models") if isinstance(data, dict) else None
    if models is None:  # tolerate the standard {"data": [...]} shape too
        models = [{"slug": m.get("id"), "visibility": "list"} for m in data.get("data", [])]
    return [
        {"name": m["slug"], "description": m.get("display_name") or m["slug"]}
        for m in models
        if m.get("slug") and m.get("visibility", "list") == "list"
    ]


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content or "")


def build_request(model: str, messages: List[BaseMessage]) -> Dict[str, Any]:
    """
    Convert LangChain messages into a Responses request that satisfies the
    ChatGPT-plan preview rules: system prompts go to ``instructions`` (system
    items are rejected), history is sent in ``input``, ``store`` is false,
    ``stream`` is true, and unsupported sampling/limit fields are omitted.
    """
    instructions: List[str] = []
    items: List[Dict[str, Any]] = []
    for message in messages:
        text = _text_of(message.content)
        if isinstance(message, SystemMessage):
            instructions.append(text)
        elif isinstance(message, AIMessage):
            items.append({"role": "assistant", "content": text})
        else:
            items.append({"role": "user", "content": text})
    if not items:
        # A prompt that is only "system" text still needs an input turn.
        items.append({"role": "user", "content": "\n\n".join(instructions)})
        instructions = []
    body: Dict[str, Any] = {
        "model": model,
        "input": items,
        "store": False,
        "stream": True,
    }
    if instructions:
        body["instructions"] = "\n\n".join(instructions)
    return body


def _sse_events(lines: Iterator[str]) -> Iterator[Dict[str, Any]]:
    data: List[str] = []
    for line in lines:
        if line == "":
            if data:
                payload = "\n".join(data)
                data = []
                if payload.strip() == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except ValueError:
                    continue
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        try:
            yield json.loads("\n".join(data))
        except ValueError:
            pass


def _event_error(event: Dict[str, Any]) -> Optional[str]:
    kind = event.get("type")
    if kind == "response.failed":
        error = (event.get("response") or {}).get("error") or {}
        return _describe_error(error.get("code"), error.get("message"))
    if kind == "response.incomplete":
        details = (event.get("response") or {}).get("incomplete_details") or {}
        return f"Response incomplete: {details.get('reason') or 'unknown reason'}"
    if kind == "error":
        return _describe_error(event.get("code"), event.get("message"))
    return None


def _describe_error(code: Optional[str], message: Optional[str]) -> str:
    if code == "subscription_sharing_usage_limit_exceeded":
        return f"ChatGPT plan usage limit reached. Manage usage at {USAGE_URL}"
    if code == "subscription_sharing_user_not_eligible":
        return "This ChatGPT account/workspace is not eligible for plan usage in other apps"
    return f"ChatGPT plan request failed: {code or ''} {message or ''}".strip()


def _http_error(status: int, body: str) -> ChatGPTPlanError:
    code = message = None
    try:
        parsed = json.loads(body)
        if isinstance(parsed.get("error"), dict):
            code = parsed["error"].get("code")
            message = parsed["error"].get("message")
        else:
            message = parsed.get("detail") or parsed.get("error")
    except (ValueError, AttributeError):
        message = body[:300]
    return ChatGPTPlanError(f"[{status}] {_describe_error(code, message)}")


class ChatGPTPlanChatModel(BaseChatModel):
    """LangChain chat model backed by the user's ChatGPT plan (Responses API)."""

    model_name: str
    access_token: SecretStr
    timeout: float = 600.0

    @property
    def _llm_type(self) -> str:
        return "chatgpt-plan"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        body = build_request(self.model_name, messages)
        completed = False
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", RESPONSES_URL, json=body, headers=self._headers()) as r:
                if r.status_code != 200:
                    raise _http_error(r.status_code, r.read().decode(errors="replace"))
                for event in _sse_events(r.iter_lines()):
                    error = _event_error(event)
                    if error:
                        raise ChatGPTPlanError(error)
                    if event.get("type") == "response.output_text.delta":
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=event.get("delta", ""))
                        )
                        if run_manager:
                            run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                        yield chunk
                    elif event.get("type") == "response.completed":
                        completed = True
                        break
        if not completed:
            raise ChatGPTPlanError("Stream ended before response.completed")

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        body = build_request(self.model_name, messages)
        completed = False
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", RESPONSES_URL, json=body, headers=self._headers()
            ) as r:
                if r.status_code != 200:
                    raise _http_error(r.status_code, (await r.aread()).decode(errors="replace"))
                data: List[str] = []
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip())
                        continue
                    if line != "" or not data:
                        continue
                    payload, data = "\n".join(data), []
                    try:
                        event = json.loads(payload)
                    except ValueError:
                        continue
                    error = _event_error(event)
                    if error:
                        raise ChatGPTPlanError(error)
                    if event.get("type") == "response.output_text.delta":
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=event.get("delta", ""))
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                        yield chunk
                    elif event.get("type") == "response.completed":
                        completed = True
                        break
        if not completed:
            raise ChatGPTPlanError("Stream ended before response.completed")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = "".join(c.text for c in self._stream(messages, stop, run_manager, **kwargs))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        parts = [c.text async for c in self._astream(messages, stop, run_manager, **kwargs)]
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="".join(parts)))]
        )


def _to_langchain_messages(messages: List[Dict[str, Any]]) -> List[BaseMessage]:
    converted: List[BaseMessage] = []
    for m in messages:
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


@dataclass
class ChatGPTPlanLanguageModel(LanguageModel):
    """
    Esperanto LanguageModel for the ChatGPT plan, so the provider fits both
    ModelManager/provision and esperanto's AIFactory (used by podcast-creator).
    ``api_key`` carries a current OAuth access token.
    """

    @property
    def provider(self) -> str:
        return PROVIDER

    def _get_default_model(self) -> str:
        return self.model_name or ""

    def _get_models(self):
        return []

    def to_langchain(self) -> ChatGPTPlanChatModel:
        return ChatGPTPlanChatModel(
            model_name=self.model_name or "",
            access_token=SecretStr(self.api_key or ""),
        )

    def _completion(self, text: str) -> ChatCompletion:
        return ChatCompletion(
            id=f"chatgpt-{uuid.uuid4().hex}",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            model=self.model_name or "",
            provider=PROVIDER,
        )

    def chat_complete(self, messages, stream=None, **kwargs):  # type: ignore[override]
        result = self.to_langchain().invoke(_to_langchain_messages(messages))
        return self._completion(_text_of(result.content))

    async def achat_complete(self, messages, stream=None, **kwargs):  # type: ignore[override]
        result = await self.to_langchain().ainvoke(_to_langchain_messages(messages))
        return self._completion(_text_of(result.content))


def build_language_model(model_name: str, access_token: str) -> ChatGPTPlanLanguageModel:
    return ChatGPTPlanLanguageModel(api_key=access_token, model_name=model_name)


def register_with_esperanto() -> None:
    """Let esperanto's AIFactory create ``chatgpt`` models (podcast-creator does)."""
    from esperanto import AIFactory

    AIFactory._provider_modules["language"][PROVIDER] = (
        f"{__name__}:ChatGPTPlanLanguageModel"
    )


register_with_esperanto()
