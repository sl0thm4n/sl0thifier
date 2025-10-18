class ModelNotInstalled(Exception):
    """Raised when the required backend is not found."""

    pass


class FaceNotDetected(Exception):
    """Raised when no face is detected in the input image."""

    pass
