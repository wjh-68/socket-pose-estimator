class ConfigError(Exception):
    pass

class QueueError(Exception):
    pass

class PacketValidationError(ValueError):
    pass

class PacketProcessingError(RuntimeError):
    pass