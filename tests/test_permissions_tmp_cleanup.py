"""Regression coverage for bounded temporary-directory cleanup permissions."""

from __future__ import annotations

import pytest


def _default_engine():
    from clawagents.tools.permissions import PermissionEngine, _DEFAULT_SECURE_RULES

    engine = PermissionEngine(default_decision="allow")
    engine.add_rules(list(_DEFAULT_SECURE_RULES))
    return engine


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /tmp/hca-split-runtime",
        "rm -r -f -- /tmp/hca-split-runtime",
        "rm /tmp/hca-split-runtime -rf",
        (
            "cd /repo4/home/xjiang2/hca_data_split/hca_pdf_splitter "
            "&& docker build -t hca-pdf-splitter:local . "
            "&& rm -rf /tmp/hca-split-runtime "
            "&& mkdir -m 700 /tmp/hca-split-runtime"
        ),
    ],
)
def test_default_permissions_allow_one_literal_tmp_descendant(command: str):
    decision, message = _default_engine().evaluate("execute", {"command": command})

    assert decision == "allow"
    assert message == ""


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf /etc/hca",
        "rm /etc/hca -rf",
        "rm -rf /home/user/hca",
        "rm -rf /tmp",
        "rm -rf /tmp/*",
        "rm -rf /tmp/hca-*",
        'rm -rf "$TMPDIR/hca"',
        'rm -rf "$HOME/hca"',
        "rm -rf ~/hca",
        "rm -rf /tmp/hca/../other",
        "rm -rf /tmp/a /tmp/b",
        "env rm -rf /tmp/hca",
    ],
)
def test_default_permissions_still_deny_unbounded_or_ambiguous_rm(command: str):
    decision, message = _default_engine().evaluate("execute", {"command": command})

    assert decision == "deny"
    assert message == "Refused destructive rm"


def test_custom_rm_deny_still_overrides_bounded_tmp_default():
    from clawagents.tools.permissions import PermissionRule

    engine = _default_engine()
    engine.add_rule(
        PermissionRule(
            tool="execute",
            arg_pattern="*rm -rf *",
            decision="deny",
            priority=200,
            message="custom deny",
        )
    )

    assert engine.evaluate(
        "execute", {"command": "rm -rf /tmp/hca-split-runtime"}
    ) == ("deny", "custom deny")
