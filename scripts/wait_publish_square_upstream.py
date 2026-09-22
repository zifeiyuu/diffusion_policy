"""Attach publication to the already-running queue without restarting its trainer."""
import json,subprocess,sys,time
from pathlib import Path
O=Path('/home/zxiao93/Documents/LaDiWM/results/dp_upstream_abs100_20260922')
while not (O/'published_commit.txt').exists():
    s=json.loads((O/'queue_status.json').read_text())
    if s['state']=='failed':raise RuntimeError(s)
    if s['state']=='complete':
        # The queue writes the final report immediately after its completion status.
        time.sleep(10)
        subprocess.run([sys.executable,'scripts/publish_square_upstream.py'],check=True)
        break
    time.sleep(45)
