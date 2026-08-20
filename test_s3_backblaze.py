import requests

# Let's test the Backblaze S3 endpoints for Faceit demos:
# Match 1: 1-e0dc53d4-0cff-45fa-bcd4-36607945eee8-1-1.dem.zst in europe-central
url1 = "https://demos-europe-central-faceit-cdn.s3.eu-central-003.backblazeb2.com/cs2/1-e0dc53d4-0cff-45fa-bcd4-36607945eee8-1-1.dem.zst"

try:
    r1 = requests.head(url1, timeout=5)
    print("EU Central S3 URL -> Status:", r1.status_code, "Headers:", dict(r1.headers))
except Exception as e:
    print("EU Central S3 URL -> Error:", e)

# Match 2: 1-3bdbc675-06c8-4594-8871-6aefb061dd18-1-5.dem.zst in us-east
for region_id in ["us-east-005", "us-east-004", "us-east-003", "us-east-002", "us-east-001"]:
    url2 = f"https://demos-us-east-faceit-cdn.s3.{region_id}.backblazeb2.com/cs2/1-3bdbc675-06c8-4594-8871-6aefb061dd18-1-5.dem.zst"
    try:
        r2 = requests.head(url2, timeout=3)
        print(f"US East S3 ({region_id}) -> Status:", r2.status_code)
        if r2.status_code == 200:
            print("FOUND WORKING S3 URL:", url2)
            break
    except Exception as e:
        print(f"US East S3 ({region_id}) -> Error:", e.__class__.__name__)
