"""Federal public holidays and transferred days off for the 2025 dataset."""

from datetime import date


# Federal holidays (Labour Code, Article 112) and transfers under Resolution
# No. 1335 of 4 October 2024: https://government.ru/docs/52895/
# Ordinary weekends are not holidays; weekday is encoded separately.
HOLIDAYS_2025 = frozenset(
    [date(2025, 1, day) for day in range(1, 9)]
    + [
        date(2025, 2, 23),
        date(2025, 3, 8),
        date(2025, 5, 1),
        date(2025, 5, 2),
        date(2025, 5, 8),
        date(2025, 5, 9),
        date(2025, 6, 12),
        date(2025, 6, 13),
        date(2025, 11, 3),
        date(2025, 11, 4),
        date(2025, 12, 31),
    ]
)


def is_holiday(day: date) -> bool:
    """Return the federal holiday flag, including transferred days off.

    Regional holidays are excluded. A new year's official transfers must be
    added before that year can be used for training or inference.
    """
    if day.year != 2025:
        raise ValueError(f"Russian holiday calendar supports 2025, got {day.year}")
    return day in HOLIDAYS_2025
