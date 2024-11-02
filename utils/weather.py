import random
import logging

logger = logging.getLogger("uvicorn.error")

def get_weather(location):
    """
    Generate a random weather report for a given location.

    Args:
        location (str): The location for which to generate the weather report.

    Returns:
        dict: A dictionary containing the location, temperature, unit, and conditions.
    """
    # Choose a random temperature and condition
    random_temperature = random.randint(50, 80)
    conditions = ["Cloudy", "Sunny", "Rainy", "Snowy", "Windy"]
    random_condition = random.choice(conditions)

    return {
        "location": location,
        "temperature": random_temperature,
        "unit": "F",
        "conditions": random_condition,
    }

