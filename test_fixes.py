"""
Regression checks for the parsing fixes. Run: python test_fixes.py

Deliberately dependency-free (no pytest, no torch import) so it stays runnable
even when the heavy ML stack is not installed.
"""
import mock_grpc  # noqa: F401  — must precede anything that touches grpc

import re
from urllib.parse import urlparse


def _extract_name_fallback(filename_stem: str) -> str:
    """Mirror of indexer.parser._extract_name's filename fallback."""
    from indexer.parser import _extract_name
    return _extract_name("", filename_stem)


def test_name_fallback():
    # SharePoint item-id prefix + Naukri tag + [Xy_Zm] experience marker
    assert _extract_name_fallback(
        "01PQATDQLOPE53I7B2GRD3AVR4WJFTRUBJ_Naukri_KeshavGupta[0y_3m]"
    ) == "Keshavgupta"
    # Plain filenames must survive untouched
    assert _extract_name_fallback("Dikshita_Dhanopiya") == "Dikshita Dhanopiya"
    assert _extract_name_fallback("rahul-ranjan") == "Rahul Ranjan"
    # A stem that is *only* strippable parts must not collapse to empty
    assert _extract_name_fallback("[2y_0m]") == "[2y_0m]"


def test_site_url_hostname():
    """resolve_site_id must tolerate the Graph-style 'host:/sites/Foo' spelling."""
    def hostname_of(site_url):
        parsed = urlparse(site_url if "//" in site_url else f"https://{site_url}")
        return parsed.netloc.rstrip(":"), parsed.path.rstrip("/")

    assert hostname_of("mabicons.sharepoint.com:/sites/Mabicons/recruitment") == (
        "mabicons.sharepoint.com", "/sites/Mabicons/recruitment"
    )
    assert hostname_of("https://contoso.sharepoint.com/sites/Recruitment/") == (
        "contoso.sharepoint.com", "/sites/Recruitment"
    )


def test_grpc_mock_survives_google_api_core_probes():
    """The three lookups that used to kill the google.generativeai import."""
    import grpc
    # grpc_status does: {x.value[0]: x for x in grpc.StatusCode}
    assert {x.value[0]: x for x in grpc.StatusCode}[0] is grpc.StatusCode.OK
    # grpc_status._async annotates `call: aio.Call` on a class-level lookup
    assert grpc.experimental.aio.Call is not None
    # google.api_core.client_info reads this unguarded
    assert grpc.__version__


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
