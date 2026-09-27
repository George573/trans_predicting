import json
import re
from pathlib import Path

import pandas as pd

ROUTES = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
LINES = Path("factors/data/route_lines.geojson")
SPRAVOCHNIK = Path("dataset/spravochniki/Хакатон_справочники_трамвай_10_маршрутов.xlsx")
OUT = Path("web/geo/routes.geojson")
DASHES = str.maketrans({chr(c): "-" for c in (0x2012, 0x2013, 0x2014, 0x2015, 0x2212)})


def clean(name: str) -> str:
    return re.sub(r'"([^"]*)"', r"«\1»", name).translate(DASHES).strip()


def osm_name(names: list[str]) -> str:
    return clean(names[0].split(": ", 1)[1].replace(" => ", " - "))


def rounded(value):
    if isinstance(value, float):
        return round(value, 6)
    return [rounded(v) for v in value]


def feature(props: dict, geometry: dict) -> dict:
    return {"type": "Feature", "properties": props, "geometry": geometry}


def main() -> None:
    lines = {f["properties"]["route"]: f for f in json.loads(LINES.read_text(encoding="utf-8"))["features"]}
    ref = pd.read_excel(SPRAVOCHNIK, sheet_name="Маршруты GTFS_ROUTES", header=1)
    names = {int(r.route_short_name): clean(r.route_long_name) for r in ref.itertuples()}
    stops = pd.read_excel(SPRAVOCHNIK, sheet_name="Порядок_с_координатами")
    out = []
    for route in ROUTES:
        line = lines[route]
        out.append(feature(
            {"kind": "line", "route": route, "name": names.get(route) or osm_name(line["properties"]["names"]),
             "source": "osm", "osm_relations": line["properties"]["osm_relations"]},
            {"type": line["geometry"]["type"], "coordinates": rounded(line["geometry"]["coordinates"])}))
    own = stops[stops["route_short_name"].isin(ROUTES)].drop_duplicates(["route_short_name", "stop_id"])
    for s in own.itertuples():
        out.append(feature(
            {"kind": "stop", "route": int(s.route_short_name), "name": clean(str(s.stop_name)), "source": "spravochnik"},
            {"type": "Point", "coordinates": [round(float(s.stop_lon), 6), round(float(s.stop_lat), 6)]}))
    assert sorted(f["properties"]["route"] for f in out if f["properties"]["kind"] == "line") == sorted(ROUTES)
    assert {f["properties"]["route"] for f in out if f["properties"]["kind"] == "stop"} == {1, 5, 7, 11, 12}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": out}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{OUT}: {len(ROUTES)} линий, {len(out) - len(ROUTES)} остановок, {OUT.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()
