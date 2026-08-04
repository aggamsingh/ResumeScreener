import enum
import sys
import types

# Custom Mock Module that inherits from types.ModuleType to act as a real Python module
class MockModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self._fallback_cache = {}

    def __getattr__(self, name):
        # Avoid recursion for standard module attributes
        if name.startswith('__'):
            raise AttributeError(name)
        if name not in self._fallback_cache:
            self._fallback_cache[name] = GenericDummy
        return self._fallback_cache[name]


class _DummyMeta(type):
    """
    Makes attribute access work on the dummy CLASS, not just its instances.

    __getattr__ above hands back the GenericDummy class itself, so any nested
    lookup landed on a class object. Instance-level __getattr__ does not apply
    there, so a chain like grpc.experimental.aio.Call raised
    "AttributeError: type object 'GenericDummy' has no attribute 'Call'" —
    which is exactly what grpc_status/_async.py does at import time, taking
    google.generativeai down with it and disabling the Gemini reranker.
    """
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return cls


class GenericDummy(metaclass=_DummyMeta):
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, key): return GenericDummy()
    def __call__(self, *args, **kwargs): return GenericDummy()

# Create the main grpc module
mock_grpc = MockModule("grpc")

# Define distinct dummy interceptor classes so subclassing works and prevents duplicate base classes
class DummyUnaryUnaryClientInterceptor: pass
class DummyUnaryStreamClientInterceptor: pass
class DummyStreamUnaryClientInterceptor: pass
class DummyStreamStreamClientInterceptor: pass

class DummyChannel:
    def __init__(self, *args, **kwargs): pass

class DummyCompression:
    NoCompression = 0
    Deflate = 1
    Gzip = 2


# grpc.StatusCode must be a real iterable enum whose members carry
# .value == (code, name). google-generativeai pulls in google.api_core, which
# imports grpc_status._common, which does:
#     {x.value[0]: x for x in grpc.StatusCode}
# Against the generic dummy that returned a *class*, this raised
# "TypeError: 'type' object is not iterable" at import time — so the entire
# google.generativeai import blew up and the Gemini reranker could never run.
# It surfaced only as "AI reranker temporarily unavailable" in every response.
class StatusCode(enum.Enum):
    OK                  = (0,  "ok")
    CANCELLED           = (1,  "cancelled")
    UNKNOWN             = (2,  "unknown")
    INVALID_ARGUMENT    = (3,  "invalid argument")
    DEADLINE_EXCEEDED   = (4,  "deadline exceeded")
    NOT_FOUND           = (5,  "not found")
    ALREADY_EXISTS      = (6,  "already exists")
    PERMISSION_DENIED   = (7,  "permission denied")
    RESOURCE_EXHAUSTED  = (8,  "resource exhausted")
    FAILED_PRECONDITION = (9,  "failed precondition")
    ABORTED             = (10, "aborted")
    OUT_OF_RANGE        = (11, "out of range")
    UNIMPLEMENTED       = (12, "unimplemented")
    INTERNAL            = (13, "internal")
    UNAVAILABLE         = (14, "unavailable")
    DATA_LOSS           = (15, "data loss")
    UNAUTHENTICATED     = (16, "unauthenticated")

# Assign required base classes and constants
mock_grpc.UnaryUnaryClientInterceptor = DummyUnaryUnaryClientInterceptor
mock_grpc.UnaryStreamClientInterceptor = DummyUnaryStreamClientInterceptor
mock_grpc.StreamUnaryClientInterceptor = DummyStreamUnaryClientInterceptor
mock_grpc.StreamStreamClientInterceptor = DummyStreamStreamClientInterceptor
mock_grpc.Compression = DummyCompression
mock_grpc.StatusCode = StatusCode
mock_grpc.RpcError = Exception
# google.api_core.client_info reads grpc.__version__ unguarded. Dunder lookups
# deliberately raise AttributeError above (to avoid recursion), so it has to be
# set explicitly or the import chain dies here instead.
mock_grpc.__version__ = "1.0.0-mock"

# Create and populate grpc.aio submodule
mock_grpc_aio = MockModule("grpc.aio")
mock_grpc_aio.ClientInterceptor = DummyUnaryUnaryClientInterceptor
mock_grpc_aio.UnaryUnaryClientInterceptor = DummyUnaryUnaryClientInterceptor
mock_grpc_aio.UnaryStreamClientInterceptor = DummyUnaryStreamClientInterceptor
mock_grpc_aio.StreamUnaryClientInterceptor = DummyStreamUnaryClientInterceptor
mock_grpc_aio.StreamStreamClientInterceptor = DummyStreamStreamClientInterceptor
mock_grpc_aio.Channel = DummyChannel

mock_grpc.aio = mock_grpc_aio

# Create and populate grpc.experimental submodule
mock_grpc_experimental = MockModule("grpc.experimental")
mock_grpc.experimental = mock_grpc_experimental

# Register mock modules in sys.modules
sys.modules['grpc'] = mock_grpc
sys.modules['grpc.aio'] = mock_grpc_aio
sys.modules['grpc.experimental'] = mock_grpc_experimental
sys.modules['grpc._cython'] = MockModule("grpc._cython")
sys.modules['grpc._cython.cygrpc'] = MockModule("grpc._cython.cygrpc")
