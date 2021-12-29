import json


#
#
#  -------- load_json -----------
#
def load_json(path: str) -> dict:
    """Load JSON configuration file."""
    with open(path) as data:
        return json.load(data)
