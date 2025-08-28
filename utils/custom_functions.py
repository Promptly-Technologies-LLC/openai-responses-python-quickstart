import random
import logging
from datetime import datetime
from typing import Sequence, Dict, Any, Callable
from openai import AsyncOpenAI
from fastapi import HTTPException


logger = logging.getLogger("uvicorn.error")


# --- Utilities for submitting tool outputs to Responses ---

async def post_tool_outputs(client: AsyncOpenAI, response_id: str, tool_call_id: str, output: str):
    try:
        submit_stream: Callable[..., Any] = getattr(client.responses, "submit_tool_outputs_stream")
        return await submit_stream(
            response_id=response_id,
            tool_outputs=[{"tool_call_id": tool_call_id, "output": output}]
        )
    except Exception as e:
        logger.error(f"Error submitting tool outputs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
                    "description": "The city and state e.g. San Francisco, CA"
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
