import requests

# URL: https://demos-us-east.backblaze.faceit-cdn.net/cs2/1-458aba12-cbb1-4823-9b75-1672e41a1050-1-1.dem.zst
# Path: /cs2/1-458aba12-cbb1-4823-9b75-1672e41a1050-1-1.dem.zst

candidates = [
    "https://demos-us-east.faceit-cdn.net/cs2/1-458aba12-cbb1-4823-9b75-1672e41a1050-1-1.dem.zst",
    "https://distribution.faceit-cdn.net/cs2/1-458aba12-cbb1-4823-9b75-1672e41a1050-1-1.dem.zst",
    "https://demos-us-east.backblazeb2.com/file/faceit-demos/cs2/1-458aba12-cbb1-4823-9b75-1672e41a1050-1-1.dem.zst",
    "https://demos-us-east.backblazeb2.com/file/demos-us-east/cs2/1-458aba12-cbb1-4823-9b75-1672e41a1050-1-1.dem.zst",
    "https://f000.backblazeb2.com/file/faceit-demos/cs2/1-458aba12-cbb1-4823-9b75-1672e41a1050-1-1.dem.zst"
]

for url in candidates:
    try:
        r = requests.head(url, timeout=3)
        print(f"[{r.status_code}] {url}")
    except Exception as e:
        print(f"[ERR] {url}: {e.__class__.__name__}")
