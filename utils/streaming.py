from typing import Optional, AsyncIterable
from dataclasses import dataclass
from openai.types.beta.threads.run import RequiredAction
from openai.types.beta.assistant_stream_event import ThreadRunRequiresAction

@dataclass
class AssistantStreamMetadata:
    """Metadata for assistant stream events that require further processing."""
    type: str  # Always "metadata"
    required_action: Optional[RequiredAction]
    step_id: str
    run_requires_action_event: Optional[ThreadRunRequiresAction]

    @classmethod
    def create(cls, 
        required_action: Optional[RequiredAction],
        step_id: str,
        run_requires_action_event: Optional[ThreadRunRequiresAction]
    ) -> "AssistantStreamMetadata":
        """Factory method to create a metadata instance with validation."""
        return cls(
            type="metadata",
            required_action=required_action,
            step_id=step_id,
            run_requires_action_event=run_requires_action_event
        )

    def requires_tool_call(self) -> bool:
        """Check if this metadata indicates a required tool call."""
        return (self.required_action is not None 
            and self.required_action.submit_tool_outputs is not None 
            and bool(self.required_action.submit_tool_outputs.tool_calls))

    def get_run_id(self) -> str:
        """Get the run ID from the requires action event, or empty string if none."""
        return self.run_requires_action_event.data.id if self.run_requires_action_event else ""


async def stream_file_content(content: bytes) -> AsyncIterable[bytes]:
    yield content