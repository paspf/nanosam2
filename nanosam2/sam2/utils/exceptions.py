from enum import Enum
class Nanosam2Error(Exception):
    class Errors(Enum):
        NoElementsIn_cond_frame_outputs = 0

    def __init__(self, message, code=None):
        super().__init__(message)  # Call the base class constructor
        self.code = code  # Optional custom attribute

    def __str__(self):
        if self.code is not None:
            return f"{self.args[0]} (Error Code: {self.code})"
        return self.args[0]