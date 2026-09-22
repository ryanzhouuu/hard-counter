"""Model workflow errors shared without coupling prediction to training modules."""


class KaggleV6ModelTrainError(ValueError):
    """Report invalid training inputs or unreadable published model artifacts."""
