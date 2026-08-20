
import requests

headers = {'Authorization': 'Bearer d095585a-df1e-40a5-9861-2eef59801ead'}
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
