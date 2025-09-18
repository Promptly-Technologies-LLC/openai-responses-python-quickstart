import random
import logging
from datetime import datetime
from typing import Sequence, Any


logger = logging.getLogger("uvicorn.error")


# --- Custom tools ---


def get_weather(location: str, dates: Sequence[str | datetime] = [datetime.today()]) -> list[dict[str, Any]]:
    """
    Example function that generates random weather reports for a given location over a date range.
    """
    weather_reports = []

    for date in dates:
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")

        # Choose a random temperature and condition
        random_temperature = random.randint(50, 80)
        conditions = ["Cloudy", "Sunny", "Rainy", "Snowy", "Windy"]
        random_condition = random.choice(conditions)

        weather_reports.append({
            "location": location,
            "date": date,
            "temperature": random_temperature,
            "unit": "F",
            "conditions": random_condition,
        })

    return weather_reports
