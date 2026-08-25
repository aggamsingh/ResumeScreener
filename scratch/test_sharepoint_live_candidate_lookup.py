import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

tenant_id = os.getenv("SHAREPOINT_TENANT_ID")
client_id = os.getenv("SHAREPOINT_CLIENT_ID")
client_secret = os.getenv("SHAREPOINT_CLIENT_SECRET")

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

# Load sample candidates from canonical_candidates.json
with open("api/canonical_candidates.json", "r", encoding="utf-8") as f:
    cands = json.load(f)

for cand in cands[:5]:
    cand_id = cand.get("id")
    name = cand.get("full_name") or cand.get("name")
    source_url = cand.get("source_file_url") or ""
    
    # Extract search term from filename or candidate name
    filename = source_url.split("/")[-1].replace(".pdf", "").replace(".docx", "") if source_url else ""
    search_term = filename if filename else name
    
    # Clean search term for Graph API search
    clean_q = search_term.split("_")[0] if "_" in search_term else search_term
    if len(clean_q) > 15:
        clean_q = clean_q[:15]
        
    print(f"SEARCHING CANDIDATE [{name}] with query [{clean_q}]...")
    
    search_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/search(q='{clean_q}')"
    s_res = requests.get(search_url, headers=headers)
    if s_res.status_code == 200:
        items = s_res.json().get("value", [])
        if items:
            best_item = items[0]
            item_id = best_item.get("id")
            content_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/content"
            c_res = requests.get(content_url, headers=headers)
            print(f"  FOUND FILE: {best_item.get('name')} | SIZE: {len(c_res.content)} bytes | PDF: {c_res.content.startswith(b'%PDF')}")
        else:
            print("  NO ITEMS FOUND IN SHAREPOINT")
    else:
        print("  SEARCH FAILED:", s_res.status_code)
    print("---")
