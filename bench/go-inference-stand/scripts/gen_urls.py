"""Список URL для cmd/load: микс запросов диспетчера.

60% маршрут за день по часам, 20% все маршруты за день, 15% маршрут за 30 дней по дням,
5% все маршруты за 30 дней по часам. python scripts/gen_urls.py > urls.txt
"""
import datetime as dt
import random

random.seed(1)
R = [1, 7, 11, 12, 17, 25, 26, 28, 50]
BASE = dt.date(2025, 11, 1)
HOST = "http://localhost:8080/api/v1/forecast?"


def d(i):
    return (BASE + dt.timedelta(days=i)).isoformat()


for _ in range(4000):
    x, k = random.random(), f"&k_weather={random.uniform(0.85, 1.1):.2f}"
    if x < 0.6:
        q = f"route={random.choice(R)}&from={d(random.randrange(61))}&granularity=hour"
    elif x < 0.8:
        q = f"from={d(random.randrange(61))}&granularity=hour"
    elif x < 0.95:
        i = random.randrange(31)
        q = f"route={random.choice(R)}&from={d(i)}&to={d(i + 29)}&granularity=day"
    else:
        i = random.randrange(30)
        q = f"from={d(i)}&to={d(i + 29)}&granularity=hour"
    print(HOST + q + k)
