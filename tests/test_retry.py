"""stream_completion retries network blips, but never repeats shown text."""

import litellm
import pytest

import agent


class _Chunk:
    usage = None

    def __init__(self, text):
        delta = type("D", (), {"content": text, "tool_calls": None})()
        self.choices = [type("C", (), {"delta": delta})()]


def _conn_error():
    return litellm.exceptions.APIConnectionError(
        message="[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac", llm_provider="openrouter",
        model="m")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)


def test_retries_connection_error_then_succeeds(monkeypatch):
    calls = []

    def fake(**kw):
        calls.append(kw)
        if len(calls) < 3:
            raise _conn_error()
        return iter([_Chunk("hello")])

    monkeypatch.setattr(litellm, "completion", fake)
    shown = []
    text, tools, _ = agent.stream_completion(shown.append, model="m", messages=[])
    assert text == "hello" and shown == ["hello"] and len(calls) == 3
    assert calls[0]["stream"] is True


def test_gives_up_after_retries(monkeypatch):
    calls = []

    def fake(**kw):
        calls.append(1)
        raise _conn_error()

    monkeypatch.setattr(litellm, "completion", fake)
    with pytest.raises(litellm.exceptions.APIConnectionError):
        agent.stream_completion(lambda t: None, model="m", messages=[])
    assert len(calls) == len(agent.RETRY_DELAYS) + 1


def test_no_retry_after_text_was_shown(monkeypatch):
    calls = []

    def stream():
        yield _Chunk("partial")
        raise _conn_error()

    def fake(**kw):
        calls.append(1)
        return stream()

    monkeypatch.setattr(litellm, "completion", fake)
    shown = []
    with pytest.raises(litellm.exceptions.APIConnectionError):
        agent.stream_completion(shown.append, model="m", messages=[])
    assert len(calls) == 1 and shown == ["partial"]  # not repeated


def test_auth_errors_are_not_retried(monkeypatch):
    calls = []

    def fake(**kw):
        calls.append(1)
        raise litellm.exceptions.AuthenticationError(message="bad key", llm_provider="x", model="m")

    monkeypatch.setattr(litellm, "completion", fake)
    with pytest.raises(litellm.exceptions.AuthenticationError):
        agent.stream_completion(lambda t: None, model="m", messages=[])
    assert len(calls) == 1
