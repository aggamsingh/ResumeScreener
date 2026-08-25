import sys
sys.path.insert(0, ".")
from api.main import app
import traceback

try:
    schema = app.openapi()
    print("OPENAPI GENERATED SUCCESSFULLY!")
    print("PATHS:")
    for path in schema["paths"]:
        print("  -", path)
except Exception as e:
    print("OPENAPI ERROR:", type(e), e)
    traceback.print_exc()
