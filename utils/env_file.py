import os
import logging


def update_env_file(var_name: str, var_value: str, logger: logging.Logger):
    """
    Update the .env file with a new environment variable.

    If the .env file already contains the specified variable, it will be updated.
    The new value will be appended to the .env file if it doesn't exist.
    If the .env file does not exist, it will be created.

    Args:
        var_name: The name of the environment variable to update
        var_value: The value to set for the environment variable
        logger: Logger instance for output
    """
    lines = []
    # Read existing contents if file exists
    if os.path.exists('.env'):
        with open('.env', 'r') as env_file:
            lines = env_file.readlines()

        # Remove any existing line with this variable
        lines = [line for line in lines if not line.startswith(f"{var_name}=")]
    else:
        # Log when we're creating a new .env file
        logger.info("Creating new .env file")

    # Sanitize value to avoid breaking lines
    sanitized_value = str(var_value).replace("\r", "").replace("\n", "")
    # Normalize existing lines to always end with a newline
    normalized_lines = [ln if ln.endswith("\n") else ln + "\n" for ln in lines]
    content = "".join(normalized_lines)
    if content and not content.endswith("\n"):
        content += "\n"
    # Write back all lines including the new variable
    with open('.env', 'w') as env_file:
        env_file.write(content)
        env_file.write(f"{var_name}={sanitized_value}\n")
    
    logger.debug(f"Environment variable {var_name} written to .env: {var_value}")
