class ConfigurationError(ValueError):
    """Raised when a manifest or run configuration is invalid."""


class DataIntegrityError(RuntimeError):
    """Raised when input data do not match their declared manifest."""
