import requests

headers = {'Authorization': 'Bearer d095585a-df1e-40a5-9861-2eef59801ead'}
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
