import logging

from clawagents.redact import redact

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

# Diagnostic logger matching the TS implementation. All emitted messages are
# routed through ``redact()`` so accidental key prints (e.g. logging an LLM
# request body that contained ``Authorization: Bearer ...``) never make it to
# the terminal or log files.
class DiagnosticLogger:
    @staticmethod
    def warn(msg: str):
        logging.warning(f"[DIAG_WARN] {redact(msg)}")

    @staticmethod
    def debug(msg: str):
        logging.debug(f"[DIAG_DEBUG] {redact(msg)}")

    @staticmethod
    def error(msg: str):
        logging.error(f"[DIAG_ERROR] {redact(msg)}")

    @staticmethod
    def info(msg: str):
        logging.info(f"[DIAG_INFO] {redact(msg)}")

diagnostic_logger = DiagnosticLogger()

def log_lane_dequeue(lane: str, waited_ms: float, queue_ahead: int):
    diagnostic_logger.debug(f"Lane {lane} dequeue. Waited {waited_ms}ms. Ahead: {queue_ahead}")

def log_lane_enqueue(lane: str, total_size: int):
    diagnostic_logger.debug(f"Lane {lane} enqueue. Total Size: {total_size}")
