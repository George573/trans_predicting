import numpy as np
import pandas as pd

from build_factors import day_off, delayed_routes, near_track, track_points

assert delayed_routes("На улице Дурова задерживаются трамваи № 7 и 50.  Маршруты временно изменены.") == {7, 50}
assert delayed_routes("На Шаболовке задерживаются трамваи № Т1, 38, 39 и 47.") == {38, 39, 47}
assert delayed_routes("На Покровском бульваре задерживаются трамваи № А, 3, 39 и 90.") == {3, 39, 90}
assert delayed_routes("В районе метро «Бабушкинская» задерживаются трамваи № 17.\nМаршрут временно изменен.") == {17}
assert delayed_routes("Трамваи 11, 17 и 25 снова ходят по своим маршрутам") == set()
assert delayed_routes("Восстановлено движение трамваев на Живописной улице.") == set()

track = track_points([[[37.60, 55.75], [37.61, 55.75]]])
assert near_track(np.array([55.7505, 55.76]), np.array([37.605, 37.605]), track, 0.1).tolist() == [True, False]

days = pd.to_datetime(["2025-10-31", "2025-11-01", "2025-11-02", "2025-11-03", "2025-11-05"])
assert day_off(pd.DatetimeIndex(days)).tolist() == [0, 0, 1, 1, 0]
print("ok")
