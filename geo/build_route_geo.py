"""Гео-признаки трамвайных маршрутов из OpenStreetMap.

Задача признаков - описать линию так, чтобы модель могла оценить её загрузку, ни разу
этой линии не видев. Поэтому здесь нет ничего, что зависит от истории посадок: только
геометрия, остановки, пересадки и то, что стоит вдоль линии.

    uv run --with requests python build_route_geo.py    # около трёх минут, дальше из кэша

Источник - OpenStreetMap через Overpass API, https://overpass-api.de/api/interpreter.
Сырые ответы кладутся в data/osm_*.json и при повторных запусках берутся с диска:
OSM живой, и без кэша числа в отчёте перестали бы воспроизводиться.

Маршрут 5 исключён: в разметке посадок его нет (он открылся 16.12.2025), проверить на
нём точность нечем, поэтому он не участвует ни в обучении, ни в оценке.
"""

import json
import math
import re
import subprocess
import time
from pathlib import Path

DATA = Path(__file__).parent / "data"
ROUTES = (1, 7, 11, 12, 17, 25, 26, 28, 50)
BBOX = "55.49,37.3,55.95,37.95"
MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

# Кремль, нулевая точка Москвы для радиальных признаков
CENTER_LON, CENTER_LAT = 37.6175, 55.7520
# локальное плоское приближение: на широте Москвы градус долготы короче градуса широты
KM_PER_DEG_LON = 111.32 * math.cos(math.radians(55.75))
KM_PER_DEG_LAT = 110.57

# Роли платформ в схеме public_transport v2. У одной физической остановки есть и platform,
# и stop_position, поэтому считать надо что-то одно, иначе остановок выйдет вдвое больше.
PLATFORM_ROLES = ("platform", "platform_entry_only", "platform_exit_only")
STOP_ROLES = ("stop", "stop_entry_only", "stop_exit_only")

TRANSFER_RADIUS_KM = 0.3      # пересадка: остановка трамвая рядом со станцией
NEAR_RADIUS_M = 500           # застройка и точки притяжения вдоль линии
BUS_RADIUS_M = 300            # подвозящая сеть
STOP_DEDUP_KM = 0.06          # две платформы ближе 60 м - одна остановка
TRACK_SNAP_M = 30             # узел остановки считается принадлежащим линии в этом радиусе

ROUTE_QUERY = (
    f'[out:json][timeout:300];relation["route"="tram"]'
    f'["ref"~"^({"|".join(str(r) for r in ROUTES)})$"]({BBOX});out geom;'
)
STATION_QUERY = (
    f'[out:json][timeout:300];('
    f'node["station"="subway"]({BBOX});'
    f'node["railway"="station"]({BBOX});'
    f'node["railway"="halt"]({BBOX});'
    f');out tags center;'
)


def tram_stops_query(relation_ids: list[int]) -> str:
    """Остановки берём узлами railway=tram_stop вдоль путей, а не членами отношения.

    Причина - неполнота разметки: в отношении маршрута 28 платформами помечены две точки
    из пятнадцати, в отношении маршрута 11 - шестнадцать при длине 16.8 км. Узлы
    railway=tram_stop лежат прямо на рельсах и размечены сплошь, поэтому счёт по ним
    воспроизводим. Платформы из отношения остаются в выводе для сверки.
    """
    ids = ",".join(str(i) for i in relation_ids)
    return (
        f"[out:json][timeout:300];rel(id:{ids});way(r)->.w;"
        f'node(around.w:{TRACK_SNAP_M})["railway"="tram_stop"];out center;'
    )


def counts_query(relation_ids: list[int]) -> str:
    """Плотность окружения вокруг линии. out count не тянет объекты, только их число."""
    ids = ",".join(str(i) for i in relation_ids)
    return (
        f"[out:json][timeout:300];rel(id:{ids});way(r)->.w;"
        f'nwr(around.w:{NEAR_RADIUS_M})["building"~"^(apartments|residential|house|dormitory)$"];out count;'
        f'nwr(around.w:{NEAR_RADIUS_M})["shop"];out count;'
        f'nwr(around.w:{NEAR_RADIUS_M})["amenity"~"^(school|university|college|kindergarten)$"];out count;'
        f'node(around.w:{BUS_RADIUS_M})["highway"="bus_stop"];out count;'
    )


def overpass(query: str) -> list[dict]:
    """Запрос через curl: он ходит через системное хранилище сертификатов, в отличие от
    urllib, который в части окружений падает на проверке цепочки.

    Зеркала перебираются по кругу с нарастающей паузой: публичные инстансы Overpass
    регулярно отвечают "server is probably too busy" вместо JSON.
    """
    for attempt, server in enumerate(MIRRORS * 3):
        result = subprocess.run(
            ["curl", "-s", "-m", "300", "-A", "tram-geo/1.0",
             "--data-urlencode", f"data={query}", server],
            capture_output=True, text=True,
        )
        try:
            return json.loads(result.stdout)["elements"]
        except (json.JSONDecodeError, KeyError):
            reason = re.sub(r"<[^>]+>", " ", result.stdout).split("Error:")[-1].strip()
            print(f"  {server}: {reason[:120] or 'нет ответа'}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("Overpass недоступен")


def cached(name: str, query: str) -> list[dict]:
    path = DATA / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print(f"качаю {name}")
    elements = overpass(query)
    path.write_text(json.dumps(elements, ensure_ascii=False), encoding="utf-8")
    return elements


def km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    return math.hypot((lon2 - lon1) * KM_PER_DEG_LON, (lat2 - lat1) * KM_PER_DEG_LAT)


def way_length_km(geometry: list[dict]) -> float:
    return sum(
        km(a["lon"], a["lat"], b["lon"], b["lat"])
        for a, b in zip(geometry, geometry[1:])
    )


def dedup_points(points: list[tuple[float, float]], radius_km: float) -> list[tuple[float, float]]:
    """жадное склеивание близких точек: платформы двух направлений - одна остановка"""
    kept: list[tuple[float, float]] = []
    for lon, lat in points:
        if not any(km(lon, lat, klon, klat) < radius_km for klon, klat in kept):
            kept.append((lon, lat))
    return kept


def parse_routes(elements: list[dict]) -> dict[int, list[dict]]:
    """маршрут -> список его направлений, каждое с путями, платформами и их геометрией"""
    directions: dict[int, list[dict]] = {route: [] for route in ROUTES}
    for relation in elements:
        route = int(relation["tags"]["ref"])
        if route not in directions:
            continue
        ways, platforms, stops = {}, [], []
        for member in relation.get("members", []):
            if member["type"] == "way" and member["role"] == "":
                ways[member["ref"]] = member.get("geometry", [])
            elif member["type"] == "node" and member["role"] in PLATFORM_ROLES:
                platforms.append((member["lon"], member["lat"]))
            elif member["type"] == "node" and member["role"] in STOP_ROLES:
                stops.append((member["lon"], member["lat"]))
        directions[route].append({
            "relation": relation["id"],
            "name": relation["tags"].get("name", ""),
            "ways": ways,
            "platforms": platforms or stops,
        })
    return directions


def parse_stations(elements: list[dict]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """станции метро и станции железной дороги, включая МЦК и МЦД"""
    subway, railway = [], []
    for element in elements:
        tags = element.get("tags", {})
        lon, lat = element.get("lon"), element.get("lat")
        if lon is None or lat is None:
            continue
        if tags.get("station") == "subway" or tags.get("subway") == "yes":
            subway.append((lon, lat))
        elif tags.get("railway") in ("station", "halt"):
            railway.append((lon, lat))
    return dedup_points(subway, 0.15), dedup_points(railway, 0.15)


def build() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    directions = parse_routes(cached("osm_routes.json", ROUTE_QUERY))
    subway, railway = parse_stations(cached("osm_stations.json", STATION_QUERY))
    print(f"станций метро {len(subway)}, железнодорожных {len(railway)}")

    # общий путь: по каким путям OSM ездит больше одного маршрута
    way_owners: dict[int, set[int]] = {}
    for route, dirs in directions.items():
        for direction in dirs:
            for way_id in direction["ways"]:
                way_owners.setdefault(way_id, set()).add(route)

    rows = []
    for route in ROUTES:
        dirs = directions[route]
        if not dirs:
            raise RuntimeError(f"маршрут {route} не найден в OSM")

        lengths = [sum(way_length_km(g) for g in d["ways"].values()) for d in dirs]
        length_km = sum(lengths) / len(lengths)

        relation_ids = [d["relation"] for d in dirs]
        tram_stops = cached(f"osm_tramstops_{route}.json", tram_stops_query(relation_ids))
        all_stops = dedup_points(
            [(n["lon"], n["lat"]) for n in tram_stops if "lon" in n], STOP_DEDUP_KM
        )
        n_stops = len(all_stops)
        n_platforms = len(dedup_points([p for d in dirs for p in d["platforms"]], STOP_DEDUP_KM))

        def near(stations: list[tuple[float, float]], radius_km: float) -> int:
            return sum(
                any(km(lon, lat, slon, slat) <= radius_km for slon, slat in stations)
                for lon, lat in all_stops
            )

        n_metro = near(subway, TRANSFER_RADIUS_KM)
        n_rail = near(railway, TRANSFER_RADIUS_KM)

        track = [(p["lon"], p["lat"]) for d in dirs for g in d["ways"].values() for p in g]
        dists = [km(lon, lat, CENTER_LON, CENTER_LAT) for lon, lat in track]
        centroid_lon = sum(lon for lon, _ in track) / len(track)
        centroid_lat = sum(lat for _, lat in track) / len(track)

        own_ways = {w for d in dirs for w in d["ways"]}
        shared_ways = {w for w in own_ways if len(way_owners[w]) > 1}
        shared_km = sum(
            way_length_km(g)
            for d in dirs for w, g in d["ways"].items() if w in shared_ways
        ) / len(dirs)
        neighbours = {r for w in shared_ways for r in way_owners[w]} - {route}

        counts = cached(f"osm_counts_{route}.json", counts_query(relation_ids))
        residential, shops, education, bus_stops = (int(c["tags"]["total"]) for c in counts)

        rows.append({
            "route": route,
            "length_km": round(length_km, 2),
            "n_stops": n_stops,
            "stop_spacing_m": round(length_km * 1000 / max(n_stops - 1, 1)),
            "n_metro_transfers": n_metro,
            "n_rail_transfers": n_rail,
            "n_metro_transfers_500m": near(subway, 0.5),
            "dist_center_km": round(km(centroid_lon, centroid_lat, CENTER_LON, CENTER_LAT), 2),
            "min_dist_center_km": round(min(dists), 2),
            "share_within_5km": round(sum(d <= 5 for d in dists) / len(dists), 3),
            "share_within_10km": round(sum(d <= 10 for d in dists) / len(dists), 3),
            "shared_track_km": round(shared_km, 2),
            "shared_share": round(shared_km / length_km, 3) if length_km else 0.0,
            "n_routes_sharing": len(neighbours),
            "n_residential_500m": residential,
            "n_shop_500m": shops,
            "n_education_500m": education,
            "n_bus_stops_300m": bus_stops,
            "n_platforms_tagged": n_platforms,
        })
        print(f"маршрут {route:>2}: {rows[-1]}")

    columns = list(rows[0])
    out = DATA / "route_geo.csv"
    out.write_text(
        ";".join(columns) + "\n"
        + "\n".join(";".join(str(row[c]) for c in columns) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"\n{out}: {len(rows)} маршрутов, {len(columns) - 1} признаков")


if __name__ == "__main__":
    build()
