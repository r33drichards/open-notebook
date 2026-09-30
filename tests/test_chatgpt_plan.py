"""Tests for Sign in with ChatGPT (ChatGPT plan usage)."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import SecretStr

from open_notebook.ai import chatgpt_plan as cp
from open_notebook.ai.provider_registry import PROVIDERS
from open_notebook.domain.credential import Credential


@pytest.fixture(autouse=True)
def _clear_pending():
    cp._pending.clear()
    yield
    cp._pending.clear()


def _query(url: str) -> dict:
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def test_provider_registered_language_only():
    spec = PROVIDERS["chatgpt"]
    assert spec.modalities == ("language",)
    assert spec.required_env == ()


def test_pkce_pair_is_s256():
    import base64
    import hashlib

    verifier, challenge = cp._pkce_pair()
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert challenge == expected
    assert len(challenge) == 43


@pytest.mark.asyncio
async def test_start_authorization_new_registration():
    with patch.object(cp, "get_host_id", AsyncMock(return_value="urn:uuid:host")):
        result = await cp.start_authorization()

    params = _query(result["authorization_url"])
    assert result["authorization_url"].startswith(cp.AUTHORIZE_URL)
    assert params["client_id"] == "dynamic_agent_client"
    assert params["agent_name_hint"] == "Open Notebook"
    assert params["ext_agent_host_id"] == "urn:uuid:host"
    assert params["redirect_uri"] == "http://127.0.0.1:1455/auth/callback"
    assert params["resource"] == "https://api.openai.com/v1"
    assert "chatgpt.tokens.use.direct" in params["scope"].split()
    assert params["code_challenge_method"] == "S256"
    assert params["state"] in cp._pending


@pytest.mark.asyncio
async def test_start_authorization_reauth_uses_issued_client():
    cred = Credential(name="ChatGPT", provider="chatgpt", modalities=["language"])
    cp._store(
        cred,
        {"access_token": "a", "refresh_token": "r", "id_token": "idt"},
        {"client_id": "oaiapp_123", "email": "u@example.com", "scopes": [cp.PLAN_SCOPE]},
    )
    with patch.object(cp, "get_host_id", AsyncMock(return_value="urn:uuid:host")), patch.object(
        Credential, "get", AsyncMock(return_value=cred)
    ):
        result = await cp.start_authorization("credential:1")

    params = _query(result["authorization_url"])
    assert params["client_id"] == "oaiapp_123"
    assert "agent_name_hint" not in params
    assert params["id_token_hint"] == "idt"
    assert params["login_hint"] == "u@example.com"
    assert "prompt" not in params


def test_parse_callback_accepts_full_url_and_query():
    url = "http://127.0.0.1:1455/auth/callback?code=c&state=s&client_id=oaiapp_1"
    assert cp.parse_callback(url) == {"code": "c", "state": "s", "client_id": "oaiapp_1"}
    assert cp.parse_callback("?code=c&state=s")["code"] == "c"
    with pytest.raises(cp.ChatGPTPlanError):
        cp.parse_callback("http://127.0.0.1:1455/auth/callback?code=c")


@pytest.mark.asyncio
async def test_complete_authorization_rejects_unknown_state():
    with pytest.raises(cp.ChatGPTPlanError, match="expired"):
        await cp.complete_authorization("http://127.0.0.1:1455/auth/callback?code=c&state=nope")


@pytest.mark.asyncio
async def test_complete_authorization_new_registration_requires_client_id():
    cp._pending["s"] = cp.PendingAuthorization(
        state="s", nonce="n", code_verifier="v", client_id=cp.DYNAMIC_CLIENT_ID,
        host_id="urn:uuid:h", credential_id=None,
    )
    with pytest.raises(cp.ChatGPTPlanError, match="client_id"):
        await cp.complete_authorization("http://127.0.0.1:1455/auth/callback?code=c&state=s")
    # State is single-use.
    assert "s" not in cp._pending


@pytest.mark.asyncio
async def test_complete_authorization_saves_credential():
    cp._pending["s"] = cp.PendingAuthorization(
        state="s", nonce="n", code_verifier="v", client_id=cp.DYNAMIC_CLIENT_ID,
        host_id="urn:uuid:h", credential_id=None,
    )
    token_response = {
        "access_token": "at", "refresh_token": "rt", "id_token": "idt",
        "expires_in": 3600,
        "scope": "chatgpt.tokens.use.direct email offline_access openid profile resource.invoke",
    }
    claims = {"sub": "user-1", "iss": cp.ISSUER, "email": "u@example.com", "nonce": "n"}
    post = AsyncMock(return_value=token_response)
    with patch.object(cp, "_post_token", post), patch.object(
        cp, "_verify_id_token", MagicMock(return_value=claims)
    ), patch.object(Credential, "get_by_provider", AsyncMock(return_value=[])), patch.object(
        Credential, "save", AsyncMock()
    ):
        cred = await cp.complete_authorization(
            "http://127.0.0.1:1455/auth/callback?code=c&state=s&client_id=oaiapp_9"
        )

    sent = post.call_args.args[0]
    assert sent["client_id"] == "oaiapp_9"
    assert sent["code_verifier"] == "v"
    assert sent["grant_type"] == "authorization_code"
    assert cred.provider == "chatgpt"
    assert cred.name == "ChatGPT (u@example.com)"
    assert cp._tokens(cred)["refresh_token"] == "rt"
    summary = cp.credential_summary(cred)
    assert summary["plan_usage_enabled"] is True
    assert summary["client_id"] == "oaiapp_9"
    # Tokens never leak into the non-secret config bag.
    assert "rt" not in json.dumps(cred.config)


def _cred(expires_at: int, scopes=(cp.PLAN_SCOPE,)) -> Credential:
    cred = Credential(name="ChatGPT", provider="chatgpt", modalities=["language"])
    cp._store(
        cred,
        {"access_token": "old", "refresh_token": "rt1", "id_token": "idt"},
        {"client_id": "oaiapp_1", "expires_at": expires_at, "scopes": list(scopes)},
    )
    return cred


@pytest.mark.asyncio
async def test_get_access_token_returns_fresh_token_without_refresh():
    cred = _cred(int(time.time()) + 3000)
    post = AsyncMock()
    with patch.object(Credential, "get", AsyncMock(return_value=cred)), patch.object(
        cp, "_post_token", post
    ):
        assert await cp.get_access_token("credential:1") == "old"
    post.assert_not_called()


@pytest.mark.asyncio
async def test_get_access_token_refreshes_and_rotates():
    cred = _cred(int(time.time()) + 10)
    post = AsyncMock(
        return_value={"access_token": "new", "refresh_token": "rt2", "expires_in": 3600,
                      "scope": cp.PLAN_SCOPE}
    )
    with patch.object(Credential, "get", AsyncMock(return_value=cred)), patch.object(
        cp, "_post_token", post
    ), patch.object(Credential, "save", AsyncMock()):
        assert await cp.get_access_token("credential:1") == "new"

    sent = post.call_args.args[0]
    assert sent == {
        "grant_type": "refresh_token", "client_id": "oaiapp_1",
        "refresh_token": "rt1", "resource": cp.RESOURCE,
    }
    tokens = cp._tokens(cred)
    assert tokens["refresh_token"] == "rt2"
    assert tokens["id_token"] == "idt"  # retained when not re-issued


@pytest.mark.asyncio
async def test_get_access_token_terminal_refresh_error_signs_out():
    cred = _cred(int(time.time()) - 10)
    err = cp.ChatGPTPlanTokenError("refresh_token_reused", {})
    with patch.object(Credential, "get", AsyncMock(return_value=cred)), patch.object(
        cp, "_post_token", AsyncMock(side_effect=err)
    ), patch.object(Credential, "save", AsyncMock()):
        with pytest.raises(cp.ChatGPTPlanError, match="sign in again"):
            await cp.get_access_token("credential:1")
    assert "access_token" not in cp._tokens(cred)


@pytest.mark.asyncio
async def test_get_access_token_requires_plan_scope():
    cred = _cred(int(time.time()) + 3000, scopes=("openid",))
    with patch.object(Credential, "get", AsyncMock(return_value=cred)):
        with pytest.raises(cp.ChatGPTPlanError, match="not authorized"):
            await cp.get_access_token("credential:1")


def test_build_request_follows_plan_rules():
    body = cp.build_request(
        "gpt-x",
        [SystemMessage("be brief"), HumanMessage("hi"), AIMessage("hello"), HumanMessage("bye")],
    )
    assert body["store"] is False and body["stream"] is True
    assert body["instructions"] == "be brief"
    assert [i["role"] for i in body["input"]] == ["user", "assistant", "user"]
    for forbidden in ("temperature", "max_output_tokens", "top_p", "previous_response_id"):
        assert forbidden not in body


def test_build_request_system_only_prompt_becomes_input():
    body = cp.build_request("gpt-x", [SystemMessage("summarize this")])
    assert body["input"] == [{"role": "user", "content": "summarize this"}]
    assert "instructions" not in body


def _sse(*events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _mock_transport(payload: bytes, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer tok"
        assert body["store"] is False and body["stream"] is True
        return httpx.Response(status, content=payload)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_chat_model_streams_until_completed():
    payload = _sse(
        {"type": "response.created"},
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        {"type": "response.completed", "response": {}},
    )
    real = httpx.AsyncClient
    with patch.object(
        cp.httpx, "AsyncClient",
        lambda **kw: real(transport=_mock_transport(payload), **kw),
    ):
        model = cp.ChatGPTPlanChatModel(model_name="gpt-x", access_token=SecretStr("tok"))
        result = await model.ainvoke("hi")
    assert result.content == "Hello"


@pytest.mark.asyncio
async def test_chat_model_surfaces_usage_limit():
    payload = _sse(
        {"type": "response.output_text.delta", "delta": "x"},
        {"type": "response.failed", "response": {"error": {
            "code": "subscription_sharing_usage_limit_exceeded", "message": "limit"}}},
    )
    real = httpx.AsyncClient
    with patch.object(
        cp.httpx, "AsyncClient",
        lambda **kw: real(transport=_mock_transport(payload), **kw),
    ):
        model = cp.ChatGPTPlanChatModel(model_name="gpt-x", access_token=SecretStr("tok"))
        with pytest.raises(cp.ChatGPTPlanError, match="usage limit"):
            await model.ainvoke("hi")


@pytest.mark.asyncio
async def test_chat_model_errors_when_stream_truncated():
    payload = _sse({"type": "response.output_text.delta", "delta": "x"})
    real = httpx.AsyncClient
    with patch.object(
        cp.httpx, "AsyncClient",
        lambda **kw: real(transport=_mock_transport(payload), **kw),
    ):
        model = cp.ChatGPTPlanChatModel(model_name="gpt-x", access_token=SecretStr("tok"))
        with pytest.raises(cp.ChatGPTPlanError, match="response.completed"):
            await model.ainvoke("hi")


def test_sync_invoke_and_esperanto_wrapper():
    payload = _sse(
        {"type": "response.output_text.delta", "delta": "ok"},
        {"type": "response.completed", "response": {}},
    )
    real = httpx.Client
    with patch.object(
        cp.httpx, "Client", lambda **kw: real(transport=_mock_transport(payload), **kw)
    ):
        lm = cp.build_language_model("gpt-x", "tok")
        assert lm.provider == "chatgpt"
        completion = lm.chat_complete([{"role": "user", "content": "hi"}])
    assert completion.content == "ok"


@pytest.mark.asyncio
async def test_list_models_filters_visibility():
    payload = {"models": [
        {"slug": "gpt-a", "display_name": "GPT A", "visibility": "list"},
        {"slug": "gpt-hidden", "display_name": "Hidden", "visibility": "hide"},
    ]}
    real = httpx.AsyncClient
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    with patch.object(cp.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw)):
        models = await cp.list_models("tok")
    assert models == [{"name": "gpt-a", "description": "GPT A"}]
