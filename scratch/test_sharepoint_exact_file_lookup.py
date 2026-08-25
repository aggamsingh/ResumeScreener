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

# Candidate 000e791b-7b80-434c-8bb9-c2791a302a69
cand = {
    "id": "000e791b-7b80-434c-8bb9-c2791a302a69",
    "full_name": "Naukri NikitaKhandelwal[5y 0m]",
    "source_file_url": "https://mabicons.sharepoint.com/sites/CVDatabase/Master%20CV/position%20wise/Naukri_NikitaKhandelwal[5y_0m]_Candidate_Profile.pdf"
}

source_url = cand.get("source_file_url") or ""
# Get filename without _Candidate_Profile.pdf
raw_filename = source_url.split("/")[-1] if "/" in source_url else cand.get("full_name")
clean_filename = raw_filename.replace("_Candidate_Profile.pdf", "").replace(".pdf", "").replace(".docx", "")

print("CLEAN FILENAME SEARCH TERM:", clean_filename)

search_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/search(q='{clean_filename}')"
s_res = requests.get(search_url, headers=headers)
print("SEARCH STATUS:", s_res.status_code)
if s_res.status_code == 200:
    items = s_res.json().get("value", [])
    print("FOUND ITEMS:", len(items))
    for item in items:
        print("  NAME:", item.get("name"))
        print("  SIZE:", item.get("size"))
        print("  WEB URL:", item.get("webUrl"))
        
        # Test content fetch
        content_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item.get('id')}/content"
        c_res = requests.get(content_url, headers=headers)
        print("  FETCH CONTENT STATUS:", c_res.status_code, "BYTES:", len(c_res.content))
        print("  IS PDF:", c_res.content.startswith(b"%PDF"))
