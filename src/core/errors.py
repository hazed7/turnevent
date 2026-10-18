class TurnEventError(Exception): ...


class ConfigError(TurnEventError): ...


class InvalidInputError(TurnEventError): ...


class DataContractError(TurnEventError): ...


class TrainingError(TurnEventError): ...


class ArtifactError(TurnEventError): ...


class SourceError(TurnEventError):
    retryable: bool

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable
