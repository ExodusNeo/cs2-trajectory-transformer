import os
import sys
import time
import pandas as pd
from demoparser2 import DemoParser

dem_path = r"cs2-trajectory-transformer/data/raw_demos/clean/1-1cfcda8f-0d0c-46ee-8863-f746235e48e7-1-1.dem"

print(f"[*] Parsing CS2 demo: {os.path.basename(dem_path)}")
parser = DemoParser(dem_path)
header = parser.parse_header()
print("\n--- DEMO HEADER ---")
for k, v in header.items():
    print(f"  {k}: {v}")

# Parse sample events
death_events = parser.parse_events(["player_death"])
if death_events and len(death_events) > 0:
    _, deaths = death_events[0]
    print(f"\n--- MATCH EVENTS ---")
    print(f"  Total kills/deaths recorded: {len(deaths)}")
    cols = [c for c in ['user_name', 'attacker_name', 'weapon', 'headshot', 'tick'] if c in deaths.columns]
    print("  Sample kill events:")
    print(deaths[cols].head(3).to_string())

# Parse sample tick trajectories for player view angles & kinematics
print(f"\n--- PARSING TRAJECTORY TICKS ---")
tick_start = time.time()
fields = ["tick", "name", "steamid", "pitch", "yaw", "X", "Y", "Z", "is_alive", "team_num"]
ticks_df = parser.parse_ticks(fields)
tick_time = time.time() - tick_start

print(f"  Total trajectory ticks parsed: {len(ticks_df):,} in {tick_time:.2f}s")
player_names = [p for p in ticks_df['name'].dropna().unique().tolist() if p]
print(f"  Unique players found ({len(player_names)}): {player_names}")

print("\n--- SAMPLE PLAYER KINEMATICS DATA ---")
if player_names:
    sample_player = player_names[0]
    sample_df = ticks_df[ticks_df['name'] == sample_player].head(5)
    print(sample_df[['tick', 'name', 'pitch', 'yaw', 'X', 'Y', 'Z', 'is_alive']].to_string())

print(f"\n[SUCCESS] Demo verified! Ready for Active Tracking Window feature extraction.")
