from collect_traffic import congestion, post_tags, telegram_page

reply = (
    '<div class="tgme_widget_message" data-post="DtOperativno/23856">'
    '<a class="tgme_widget_message_reply" href="https://t.me/DtOperativno/23855">'
    '<div class="tgme_widget_message_text js-message_reply_text">На МКАД произошло ДТП.</div></a>'
    '<div class="tgme_widget_message_text js-message_text" dir="auto">Движение<br/>восстановлено.</div>'
    '<time datetime="2025-11-29T17:57:59+00:00" class="time">17:57</time></div>'
)
[(post_id, moment, text)] = telegram_page(reply)
assert (post_id, moment.isoformat(), text) == (23856, "2025-11-29T20:57:59+03:00", "Движение\nвосстановлено.")

assert congestion("По данным ЦОДД, сейчас в городе 5 баллов. Вечером ожидается до 8 баллов.")[:2] == [5, 8]
assert congestion("На дорогах - 6 баллов. Средняя скорость движения в городе - 24 км/ч.") == [6, None, 24, None, None]
assert congestion(
    "🚦Загруженность дорог - 3 балла\n🚘 На 08:50 на автомобильных дорогах МО зафиксировано 851 946 авт., "
    "что на 1% меньше по сравнению со средними значениями прошлого месяца"
) == [3, None, None, 851946, -1.0]
assert congestion("По прогнозу ЦОДД, сегодня вечером на дорогах ожидается 8-9 баллов.") == [None, 9, None, None, None]
assert congestion("1️⃣1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣ На дорогах - 7 баллов по данным ЦОДД.")[0] == 7
assert congestion("Ветер 12 м/с, порывы до 7 баллов по шкале Бофорта.") is None
assert congestion("Трамваи 11, 17 и 25 снова ходят по своим маршрутам") is None
assert post_tags("Движение трамваев временно закрыто из-за ДТП") == "closure|tram|accident"
assert post_tags("Движение по внутренней стороне МКАД восстановлено.") == "reopen"
print("ok")
