import os
import requests
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

# Search for Nikitakhandelwal in drive
search_term = "NikitaKhandelwal"
search_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/search(q='{search_term}')"

res = requests.get(search_url, headers=headers)
print("SEARCH FILE STATUS:", res.status_code)
if res.status_code == 200:
    items = res.json().get("value", [])
    print("FOUND SEARCH RESULTS:", len(items))
    for item in items:
        print("  FILE NAME:", item.get("name"))
        print("  ITEM ID:", item.get("id"))
        print("  SIZE:", item.get("size"))
        print("  WEB URL:", item.get("webUrl"))
        
        # Test downloading file content bytes!
        item_id = item.get("id")
        content_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/content"
        content_res = requests.get(content_url, headers=headers)
        print("  CONTENT FETCH STATUS:", content_res.status_code, "BYTES READ:", len(content_res.content))
        print("  STARTS WITH %PDF:", content_res.content.startswith(b"%PDF"))
        print("---")
