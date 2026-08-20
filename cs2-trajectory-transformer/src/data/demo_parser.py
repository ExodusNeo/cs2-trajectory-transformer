"""
CS2 Replay (.dem) Telemetry Parser Module.
Wraps demoparser2 to extract high-frequency tick states, player view coordinates,
and combat event timestamps.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union

try:
    from demoparser2 import DemoParser
except ImportError:
    DemoParser = None


DEFAULT_WANTED_PROPS = [
    "tick",
    "steamid",
    "name",
    "team_num",
    "X",
    "Y",
    "Z",
    "pitch",
    "yaw",
    "health",
    "armor_value",
    "is_alive",
    "is_blind",
    "active_weapon_name"
]

DEFAULT_COMBAT_EVENTS = [
    "weapon_fire",
    "player_hurt",
    "player_death",
    "player_blind"
]


class CS2DemoParser:
    """
    High-performance parser for CS2 replay files (.dem).
    Extracts per-tick player coordinate streams and game events.
    """
    def __init__(self, demo_path: str):
        if DemoParser is None:
            raise ImportError(
                "demoparser2 is not installed. Run `pip install demoparser2`."
            )
        if not os.path.exists(demo_path):
            raise FileNotFoundError(f"Demo file not found at: {demo_path}")
            
        self.demo_path = demo_path
        self._parser = DemoParser(demo_path)
        self.header = None
        
    def parse_header(self) -> Dict[str, Union[str, int, float]]:
        """Parses map name, tick rate, and match duration from demo header."""
        if self.header is None:
            self.header = self._parser.parse_header()
        return self.header
        
    def parse_ticks(
        self, 
        wanted_props: Optional[List[str]] = None,
        ticks: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Extracts tick-by-tick telemetry across all players in the match.
        """
        props = wanted_props or DEFAULT_WANTED_PROPS
        if ticks:
            df = self._parser.parse_ticks(wanted_props=props, ticks=ticks)
        else:
            df = self._parser.parse_ticks(wanted_props=props)
            
        if isinstance(df, pd.DataFrame):
            return df
        return pd.DataFrame(df)
        
    def parse_events(
        self, 
        event_names: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Extracts combat and weapon event tables (weapon_fire, player_hurt, player_death).
        """
        events_to_parse = event_names or DEFAULT_COMBAT_EVENTS
        events_dict = {}
        
        for evt in events_to_parse:
            try:
                evt_data = self._parser.parse_event(evt)
                events_dict[evt] = pd.DataFrame(evt_data) if not isinstance(evt_data, pd.DataFrame) else evt_data
            except Exception as e:
                # Event might not exist in this demo
                events_dict[evt] = pd.DataFrame()
                
        return events_dict

    def extract_player_telemetry(
        self, 
        steamid: int, 
        wanted_props: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extracts and filters tick data for a single specific player.
        """
        df_all = self.parse_ticks(wanted_props=wanted_props)
        player_df = df_all[df_all['steamid'] == steamid].sort_values('tick').reset_index(drop=True)
        return player_df
