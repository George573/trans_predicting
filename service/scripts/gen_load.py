import datetime as dt
import json
import random

ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
BASE = dt.date(2025, 11, 1)


def day(i):
    return (BASE + dt.timedelta(days=i)).isoformat()


def rain(scope):
    return {"id": "c1", "type": "rain", "value": 1.2, "scope": scope}


def body():
    x = random.random()
    req = {"model": random.choice(["catboost", "cnn"]), "corridor": True, "conditions": []}
    if x < 0.8:
        d = day(random.randrange(61))
        req |= {"from": d, "to": d, "granularity": "hour", "horizon": "day"}
        if x < 0.6:
            r = random.choice(ROUTES)
            req["routes"] = [r]
            scope = {"routes": [r], "hours": [16, 22]}
        else:
            scope = {"hours": [16, 22]}
        if random.random() < 0.5:
            req["conditions"] = [rain(scope)]
    elif x < 0.95:
        i = random.randrange(32)
        req |= {"routes": [random.choice(ROUTES)], "from": day(i), "to": day(i + 29), "granularity": "day", "horizon": "month"}
    else:
        i = random.randrange(32)
        req |= {"from": day(i), "to": day(i + 29), "granularity": "hour", "horizon": "month"}
    return req


random.seed(1)
for _ in range(4000):
    print(json.dumps(body(), separators=(",", ":")))
