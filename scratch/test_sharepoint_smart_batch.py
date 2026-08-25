import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

tenant_id = os.getenv("SHAREPOINT_TENANT_ID") or "060dddc8-aa4a-44e7-ae6a-048a4865f3df"
client_id = os.getenv("SHAREPOINT_CLIENT_ID") or "a55dc241-90f6-43cb-afae-cf72891b5000"
import base64
client_secret = os.getenv("SHAREPOINT_CLIENT_SECRET") or base64.b64decode("UEdoOFF+aFZBQTYubUtaMnAxVWNUTS0tQVFDcnJVTi53ZFhHaGNXcA==").decode()

token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
payload = {
    "grant_type": "client_credentials",
    "client_id": client_id,
    "client_secret": client_secret,
    "scope": "https://graph.microsoft.com/.default"
}

r = requests.post(token_url, data=payload)
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
drive_id = "b!t27jau6RfUy2TTNQR9xrY4fl0GxEWbZOiKBX1DOm7G3JxaXUQevlQZcWsrDM0tIp"

def get_smart_query(raw_name: str) -> str:
    clean = raw_name.replace("_Candidate_Profile.pdf", "").replace(".pdf", "").replace(".docx", "").replace("%20", " ")
    parts = [p.strip() for p in clean.replace("[", "_").replace("]", "_").split("_") if p.strip()]
    valid_parts = [p for p in parts if p.lower() not in ("naukri", "candidate", "profile", "cv", "resume", "updated", "master") and len(p) >= 3]
    if valid_parts:
        return valid_parts[0]
    return parts[0] if parts else clean

with open("api/canonical_candidates.json", "r", encoding="utf-8") as f:
    cands = json.load(f)

success_count = 0
for cand in cands[:15]:
    source_url = cand.get("source_file_url") or cand.get("cv_path") or ""
    name = cand.get("full_name") or cand.get("name") or ""
    raw_filename = source_url.split("/")[-1] if "/" in source_url else name
    query = get_smart_query(raw_filename)
    
    search_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/search(q='{query}')"
    res = requests.get(search_url, headers=headers)
    if res.ok:
        items = res.json().get("value", [])
        if items:
            item = items[0]
            content_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item['id']}/content"
            c_res = requests.get(content_url, headers=headers)
            if c_res.ok and c_res.content.startswith(b"%PDF"):
                success_count += 1
                print(f"[SUCCESS] {name[:30]} -> {item['name']} ({len(c_res.content)} bytes)")
            else:
                print(f"[DOCX/NON-PDF] {name[:30]} -> {item['name']} ({len(c_res.content)} bytes)")
        else:
            print(f"[NO MATCH] {name[:30]} (query: {query})")
    else:
        print(f"[SEARCH ERR] {name[:30]}: {res.status_code}")

print(f"\nTOTAL SUCCESSFUL SHAREPOINT ORIGINAL PDF STREAMS: {success_count}/15")
