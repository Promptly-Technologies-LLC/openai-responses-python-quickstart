import random
import logging
from datetime import datetime
from typing import Sequence, Dict, Any


logger = logging.getLogger("uvicorn.error")


# --- Custom tools ---

def get_function_tool_def() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "get_weather",
        "description": "Determine weather in my location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., 'San Francisco, CA'"
                },
                "dates": {
                    "type": "array",
                    "description": "The dates (\"YYYY-MM-DD\") to get weather for",
                    "items": {"type": "string"}
                }
            },
            "required": ["location"],
            "additionalProperties": False
        },
        "strict": False
    }


def get_weather(location, dates: Sequence[str | datetime] = [datetime.today()]) -> list[dict[str, Any]]:
    """
    Generate random weather reports for a given location over a date range.

    Args:
        location (str): The location for which to generate the weather report.
        start_date (datetime, optional): The start date for the weather report range.
        end_date (datetime, optional): The end date for the weather report range. Defaults to today.

    Returns:
        list: A list of dictionaries, each containing the location, date, temperature, unit, and conditions.
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
