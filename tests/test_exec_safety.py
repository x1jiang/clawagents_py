"""Regression tests for the exec tool's dangerous-command denylist.

Pinpoints behavior of `_is_dangerous_command`:
  - false positives (legitimate idioms like `> /dev/null`) must NOT be blocked
  - genuine destructive patterns (e.g. dd to a block device) must STILL be blocked
"""

from clawagents.tools.exec import _is_dangerous_command, BLOCKED_PATTERNS


class TestExecDenylist:
    def test_redirect_to_dev_null_is_allowed(self):
        # Common idiom — should NOT be flagged.
        assert _is_dangerous_command("ls > /dev/null") is False
        assert _is_dangerous_command("make 2>&1 > /dev/null") is False
        assert _is_dangerous_command("noisy_cmd > /dev/null 2>&1") is False

    def test_dd_to_block_device_still_blocked(self):
        assert _is_dangerous_command("dd if=/dev/zero of=/dev/sda") is True

    def test_rm_rf_root_still_blocked(self):
        assert _is_dangerous_command("rm -rf /") is True
        assert _is_dangerous_command("sudo rm -rf /") is True

    def test_fork_bomb_still_blocked(self):
        assert _is_dangerous_command(":(){ :|:& };:") is True

    def test_dev_null_not_in_blocked_patterns_list(self):
        # Defense-in-depth: ensures the regression doesn't sneak back in.
        assert "> /dev/null" not in BLOCKED_PATTERNS

    def test_dev_sd_caught_by_dangerous_re(self):
        # Block-device redirect protection moved from BLOCKED_PATTERNS
        # (substring match) to _DANGEROUS_RE + the bash validator.
        assert _is_dangerous_command("dd if=/dev/zero of=/dev/sda") is True
        assert _is_dangerous_command("echo x > /dev/sda") is True
        assert _is_dangerous_command("echo x >'/dev/sda'") is True
