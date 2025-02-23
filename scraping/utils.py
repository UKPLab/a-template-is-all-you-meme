import time
import requests
import json


def request_get_snapshot(url):
    try:
        response = requests.get(f'http://archive.org/wayback/available?url={url}')
    except:
        time.sleep(60*10)
        response = requests.get(f'http://archive.org/wayback/available?url={url}')
    print(response)
    try:
        data = response.json()
    except:
        return None
    
    if 'closest' in data['archived_snapshots']:
        snapshot = data['archived_snapshots']['closest']
        assert snapshot['available'] is True, f'Snapshot not available: {url}'
        assert snapshot['url'] is not None, f'Snapshot URL is None: {url}'
        return snapshot
    return None
