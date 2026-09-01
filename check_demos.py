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
rankings = requests.get('https://open.faceit.com/data/v4/rankings/games/cs2/regions/EU?limit=5', headers=headers).json()
for p in rankings.get('items', []):
    p_id = p.get('player_id')
    nick = p.get('nickname')
    hist = requests.get(f'https://open.faceit.com/data/v4/players/{p_id}/history?game=cs2&limit=1', headers=headers).json()
    for m in hist.get('items', []):
        m_id = m.get('match_id')
        m_res = requests.get(f'https://open.faceit.com/data/v4/matches/{m_id}', headers=headers).json()
        d_urls = m_res.get('demo_url', [])
        print(f'{nick} -> Match: {m_id} -> Demo: {d_urls}')
