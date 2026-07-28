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
            # Return a generic dummy object/class
            class GenericDummy:
                def __init__(self, *args, **kwargs): pass
                def __getattr__(self, key): return GenericDummy()
                def __call__(self, *args, **kwargs): return GenericDummy()
            self._fallback_cache[name] = GenericDummy
        return self._fallback_cache[name]

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

# Assign required base classes and constants
mock_grpc.UnaryUnaryClientInterceptor = DummyUnaryUnaryClientInterceptor
mock_grpc.UnaryStreamClientInterceptor = DummyUnaryStreamClientInterceptor
mock_grpc.StreamUnaryClientInterceptor = DummyStreamUnaryClientInterceptor
mock_grpc.StreamStreamClientInterceptor = DummyStreamStreamClientInterceptor
mock_grpc.Compression = DummyCompression
mock_grpc.RpcError = Exception

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
