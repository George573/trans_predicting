"""Проверка разбора OSM и геометрии. Сеть не нужна, все входы синтетические."""

from build_route_geo import (
    dedup_points, km, parse_routes, parse_stations, unique_stops, way_length_km,
)

# градус долготы на широте Москвы около 62.6 км, градус широты около 110.6 км
assert round(km(37.6, 55.75, 38.6, 55.75), 1) == 62.7
assert round(km(37.6, 55.75, 37.6, 56.75), 1) == 110.6
assert km(37.6, 55.75, 37.6, 55.75) == 0.0

line = [{"lon": 37.60, "lat": 55.75}, {"lon": 37.61, "lat": 55.75}, {"lon": 37.62, "lat": 55.75}]
assert round(way_length_km(line), 3) == round(2 * km(37.60, 55.75, 37.61, 55.75), 3)
assert way_length_km([{"lon": 37.6, "lat": 55.75}]) == 0.0

# склейка близких точек: вторая в 20 м от первой, третья в километре
assert dedup_points([(37.600, 55.750), (37.6003, 55.750), (37.615, 55.750)], 0.15) == [
    (37.600, 55.750), (37.615, 55.750),
]

# остановки склеиваются по названию: четыре узла двух направлений дают две остановки
nodes = [
    {"lon": 37.60, "lat": 55.75, "tags": {"name": "Школьный парк"}},
    {"lon": 37.6005, "lat": 55.7502, "tags": {"name": "Школьный парк"}},
    {"lon": 37.61, "lat": 55.75, "tags": {"name": "Поликлиника"}},
    {"lon": 37.6102, "lat": 55.7501, "tags": {"name": "Поликлиника"}},
    {"lat": 55.76, "tags": {"name": "без координат"}},
]
assert len(unique_stops(nodes)) == 2
# безымянные узлы не теряются, а склеиваются по расстоянию
assert len(unique_stops(nodes + [{"lon": 37.70, "lat": 55.75, "tags": {}},
                                 {"lon": 37.7001, "lat": 55.75, "tags": {}}])) == 3

# платформы берутся по роли, stop_position идёт в дело только если платформ нет вовсе
relations = [
    {"id": 1, "tags": {"ref": "17", "name": "Трамвай 17: туда"}, "members": [
        {"type": "way", "ref": 100, "role": "", "geometry": line},
        {"type": "node", "ref": 200, "role": "platform", "lon": 37.60, "lat": 55.75},
        {"type": "node", "ref": 201, "role": "stop", "lon": 37.60, "lat": 55.75},
        {"type": "way", "ref": 101, "role": "platform", "geometry": line},
    ]},
    {"id": 2, "tags": {"ref": "25", "name": "Трамвай 25: туда"}, "members": [
        {"type": "way", "ref": 100, "role": "", "geometry": line},
        {"type": "node", "ref": 202, "role": "stop", "lon": 37.61, "lat": 55.75},
    ]},
]
parsed = parse_routes(relations)
assert list(parsed[17][0]["ways"]) == [100]
assert parsed[17][0]["platforms"] == [(37.60, 55.75)]        # роль platform, не stop
assert parsed[25][0]["platforms"] == [(37.61, 55.75)]        # платформ нет, пошли stop
assert parsed[1] == [] and 5 not in parsed                   # маршрут 5 из перечня исключён

subway, railway = parse_stations([
    {"lon": 37.60, "lat": 55.75, "tags": {"station": "subway", "railway": "station"}},
    {"lon": 37.6005, "lat": 55.7502, "tags": {"station": "subway"}},   # тот же вестибюль
    {"lon": 37.70, "lat": 55.80, "tags": {"railway": "halt"}},
    {"lon": 37.80, "lat": 55.90, "tags": {"railway": "station"}},
    {"lat": 55.90, "tags": {"railway": "station"}},                    # без координат
])
assert len(subway) == 1 and len(railway) == 2

print("ok")
