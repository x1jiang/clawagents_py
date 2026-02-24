import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

# Diagnostic logger matching the TS implementation
class DiagnosticLogger:
    @staticmethod
    def warn(msg: str):
        logging.warning(f"[DIAG_WARN] {msg}")

    @staticmethod
    def debug(msg: str):
        logging.debug(f"[DIAG_DEBUG] {msg}")

    @staticmethod
    def error(msg: str):
        logging.error(f"[DIAG_ERROR] {msg}")

    @staticmethod
    def info(msg: str):
        logging.info(f"[DIAG_INFO] {msg}")

diagnostic_logger = DiagnosticLogger()

def log_lane_dequeue(lane: str, waited_ms: float, queue_ahead: int):
    diagnostic_logger.debug(f"Lane {lane} dequeue. Waited {waited_ms}ms. Ahead: {queue_ahead}")

def log_lane_enqueue(lane: str, total_size: int):
    diagnostic_logger.debug(f"Lane {lane} enqueue. Total Size: {total_size}")
