import os
import sys
import requests


def get_api_key():
    for p in ['.env', 'cs2-trajectory-transformer/.env', os.path.join(os.path.dirname(__file__), '.env')]:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        os.environ.setdefault(k.strip(), v.strip())
    key = os.environ.get('FACEIT_API_KEY')
    if not key:
        print("Error: FACEIT_API_KEY environment variable not set. Please create a .env file.")
        sys.exit(1)
    return key


headers = {'Authorization': f'Bearer {get_api_key()}'}
regions = ['US', 'NA', 'SA', 'SEA', 'OCE', 'EU']
for r in regions:
    try:
        rank = requests.get(f'https://open.faceit.com/data/v4/rankings/games/cs2/regions/{r}?limit=1', headers=headers).json()
        items = rank.get('items', [])
        if items:
            p_id = items[0].get('player_id')
            hist = requests.get(f'https://open.faceit.com/data/v4/players/{p_id}/history?game=cs2&limit=1', headers=headers).json()
            h_items = hist.get('items', [])
            if h_items:
                m_id = h_items[0].get('match_id')
                m_res = requests.get(f'https://open.faceit.com/data/v4/matches/{m_id}', headers=headers).json()
                print(f'Region {r} -> Demo: {m_res.get("demo_url")}')
    except Exception as e:
        print(f'Region {r} err: {e}')
