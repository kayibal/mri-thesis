import numpy as np
from scipy import stats
import soundcloud

client = soundcloud.Client(client_id="5d6b59a74f85a1878139f915b428b23a")

aids = []
artists = []
for aid in aids:
    artists.append(client.get("/artists" id=aid))
    