import random
import logging
from datetime import datetime
from typing import Sequence
from openai import AsyncOpenAI
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from pydantic import BaseModel
from typing import Dict, Any
from fastapi import HTTPException
from openai.lib.streaming._assistants import AsyncAssistantStreamManager
from openai.types.beta.function_tool_param import FunctionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition


logger = logging.getLogger("uvicorn.error")


# --- Utilities for submitting tool outputs to the assistant ---


class ToolCallOutputs(BaseModel):
    tool_outputs: Dict[str, Any]
    runId: str

async def post_tool_outputs(client: AsyncOpenAI, data: Dict[str, Any], thread_id: str) -> AsyncAssistantStreamManager:
    """
    data is expected to be something like
    {
      "tool_outputs": {
        "output": [{"location": "City", "temperature": 70, "conditions": "Sunny"}],
        "tool_call_id": "call_123"
      },
      "runId": "some-run-id",
    }
    """
    try:
        outputs_list = [
            ToolOutput(
                output=str(data["tool_outputs"]["output"]),
                tool_call_id=data["tool_outputs"]["tool_call_id"]
            )
        ]


        stream_manager = client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=thread_id,
            run_id=data["runId"],
            tool_outputs=outputs_list,
        )

        return stream_manager

    except Exception as e:
        logger.error(f"Error submitting tool outputs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Custom tools ---

def get_function_tool_param() -> FunctionToolParam:
    return FunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name="get_weather",
            description="Determine weather in my location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state e.g. San Francisco, CA"
                    },
                    "dates": {
                        "type": "array",
                        "description": "The dates (\"YYYY-MM-DD\") to get weather for",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            strict=False
        )
    )


def get_weather(location, dates: Sequence[str | datetime] = [datetime.today()]):
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
