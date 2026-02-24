from enum import Enum

class CommandLane(str, Enum):
    Main = "main"
    Cron = "cron"
    Subagent = "subagent"
    Nested = "nested"
