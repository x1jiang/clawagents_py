"""Tests for clawagents.aux_models."""

from __future__ import annotations

import pytest

from clawagents.aux_models import (
    AuxModelRegistry,
    AuxModelSpec,
    AuxModelTask,
)


def test_coerce_string_model():
    spec = AuxModelSpec.coerce("gpt-5.4-mini")
    assert spec.model == "gpt-5.4-mini"
    assert spec.base_url is None


def test_coerce_model_at_base_url():
    spec = AuxModelSpec.coerce("llama3.2:3b@http://localhost:11434")
    assert spec.model == "llama3.2:3b"
    assert spec.base_url == "http://localhost:11434"


def test_coerce_passthrough_existing_spec():
    src = AuxModelSpec(model="gpt-5.4")
    out = AuxModelSpec.coerce(src)
    assert out is src


def test_coerce_rejects_empty():
    with pytest.raises(ValueError):
        AuxModelSpec.coerce("")
    with pytest.raises(ValueError):
        AuxModelSpec.coerce("   ")


def test_with_overrides_returns_new_instance():
    spec = AuxModelSpec(model="gpt-5.4")
    out = spec.with_overrides(max_tokens=20, temperature=0.0)
    assert out is not spec
    assert out.model == "gpt-5.4"
    assert out.max_tokens == 20
    assert out.temperature == 0.0
    # original is untouched (frozen)
    assert spec.max_tokens is None


def test_registry_primary_required_and_returned():
    reg = AuxModelRegistry("gpt-5.4")
    assert reg.primary().model == "gpt-5.4"
    assert reg.get(AuxModelTask.PRIMARY).model == "gpt-5.4"


def test_registry_falls_back_to_primary():
    reg = AuxModelRegistry("gpt-5.4")
    # No COMPRESSION binding → fall back to PRIMARY.
    spec = reg.get(AuxModelTask.COMPRESSION)
    assert spec.model == "gpt-5.4"
    assert reg.has(AuxModelTask.COMPRESSION) is False


def test_registry_set_and_get_aux_task():
    reg = AuxModelRegistry("gpt-5.4")
    reg.set(AuxModelTask.COMPRESSION, "gpt-5.4-mini")
    spec = reg.get(AuxModelTask.COMPRESSION)
    assert spec.model == "gpt-5.4-mini"
    assert reg.has(AuxModelTask.COMPRESSION) is True
    # Other tasks still fall back.
    assert reg.get(AuxModelTask.TITLE).model == "gpt-5.4"


def test_registry_set_with_spec_object():
    reg = AuxModelRegistry("gpt-5.4")
    reg.set(AuxModelTask.TITLE, AuxModelSpec(model="gpt-5.4-mini", max_tokens=20))
    spec = reg.get(AuxModelTask.TITLE)
    assert spec.model == "gpt-5.4-mini"
    assert spec.max_tokens == 20


def test_registry_overwrite_binding():
    reg = AuxModelRegistry("gpt-5.4")
    reg.set(AuxModelTask.COMPRESSION, "a")
    reg.set(AuxModelTask.COMPRESSION, "b")
    assert reg.get(AuxModelTask.COMPRESSION).model == "b"


def test_registry_unset_binding():
    reg = AuxModelRegistry("gpt-5.4")
    reg.set(AuxModelTask.COMPRESSION, "gpt-5.4-mini")
    reg.unset(AuxModelTask.COMPRESSION)
    assert reg.has(AuxModelTask.COMPRESSION) is False
    assert reg.get(AuxModelTask.COMPRESSION).model == "gpt-5.4"  # falls back


def test_registry_unset_primary_raises():
    reg = AuxModelRegistry("gpt-5.4")
    with pytest.raises(ValueError):
        reg.unset(AuxModelTask.PRIMARY)


def test_registry_accepts_custom_string_task():
    reg = AuxModelRegistry("gpt-5.4")
    reg.set("embedding", "text-embedding-3-large")
    assert reg.has("embedding")
    assert reg.get("embedding").model == "text-embedding-3-large"
    assert reg.get("not-set").model == "gpt-5.4"  # fallback


def test_registry_slots_returns_copy():
    reg = AuxModelRegistry("gpt-5.4")
    reg.set(AuxModelTask.COMPRESSION, "gpt-5.4-mini")
    snap = reg.slots()
    snap.clear()  # mutating the snapshot must not affect the registry
    assert reg.has(AuxModelTask.COMPRESSION)


def test_from_env_pulls_known_aux_vars():
    env = {
        "CLAW_MODEL_COMPRESSION": "gpt-5.4-mini",
        "CLAW_MODEL_TITLE": "claude-4.5-haiku@http://h",
        "CLAW_MODEL_VISION": "gemini-3.1-pro",
        "CLAW_MODEL_JUDGE": "  ",  # blank → ignored
        "UNRELATED_VAR": "ignored",
    }
    reg = AuxModelRegistry.from_env("gpt-5.4", env=env)
    assert reg.get(AuxModelTask.COMPRESSION).model == "gpt-5.4-mini"
    title = reg.get(AuxModelTask.TITLE)
    assert title.model == "claude-4.5-haiku"
    assert title.base_url == "http://h"
    assert reg.get(AuxModelTask.VISION).model == "gemini-3.1-pro"
    # JUDGE was blank → falls back to primary.
    assert reg.has(AuxModelTask.JUDGE) is False
    assert reg.get(AuxModelTask.JUDGE).model == "gpt-5.4"


def test_from_env_defaults_to_os_environ(monkeypatch):
    monkeypatch.setenv("CLAW_MODEL_COMPRESSION", "gpt-5.4-mini")
    monkeypatch.delenv("CLAW_MODEL_TITLE", raising=False)
    monkeypatch.delenv("CLAW_MODEL_VISION", raising=False)
    monkeypatch.delenv("CLAW_MODEL_JUDGE", raising=False)
    reg = AuxModelRegistry.from_env("gpt-5.4")
    assert reg.get(AuxModelTask.COMPRESSION).model == "gpt-5.4-mini"
    assert reg.get(AuxModelTask.TITLE).model == "gpt-5.4"  # fallback


def test_from_env_with_aux_spec_primary():
    primary = AuxModelSpec(model="gpt-5.4", base_url="http://gw")
    reg = AuxModelRegistry.from_env(primary, env={})
    assert reg.primary() is primary
    assert reg.get(AuxModelTask.COMPRESSION) is primary  # fallback by identity
