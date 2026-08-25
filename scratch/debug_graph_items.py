import requests, base64

t = "060dddc8-aa4a-44e7-ae6a-048a4865f3df"
c = "a55dc241-90f6-43cb-afae-cf72891b5000"
s = base64.b64decode("UEdoOFF+aFZBQTYubUtaMnAxVWNUbS0tQVFDcnJVTi53ZFhHaGNXcA==").decode()

r = requests.post(f"https://login.microsoftonline.com/{t}/oauth2/v2.0/token", data={"grant_type": "client_credentials", "client_id": c, "client_secret": s, "scope": "https://graph.microsoft.com/.default"})
token = r.json()["access_token"]

drive_id = "b!t27jau6RfUy2TTNQR9xrY4fl0GxEWbZOiKBX1DOm7G3JxaXUQevlQZcWsrDM0tIp"
headers = {"Authorization": f"Bearer {token}"}
search_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/search(q='NikitaKhandelwal')"

res = requests.get(search_url, headers=headers)
items = res.json().get("value", [])
print(f"SEARCH RETURNED {len(items)} ITEMS:")
for i, item in enumerate(items):
    print(f"[{i}] NAME: {item.get('name')} | SIZE: {item.get('size')} | WEB_URL: {item.get('webUrl')}")
