

class SampleEmptyLimitError(SampleEmptyError):
    """Exception raised when SampleEmptyError is raised
    continuously for some specific limit."""

    def __init__(
        self,
        message="Generated sample is still empty after the set limit.",
        in_size=None,
    ):
        super().__init__(message, in_size)
