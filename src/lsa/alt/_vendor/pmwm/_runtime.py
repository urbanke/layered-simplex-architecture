"""Private settings: ambient PMM_* environment never changes ALT calculations."""
from contextvars import ContextVar
import os as _os

_settings = ContextVar("alt_pmwm_settings", default={})

class _Environment:
    def get(self, key, default=None):
        return _settings.get().get(key, default)

environ = _Environment()

def __getattr__(name):
    return getattr(_os, name)
