from dataclasses import dataclass, field


@dataclass
class ToolTaskResult:
    """Result of a single tool execution task."""
    sse_events: list[tuple[str, str]] = field(default_factory=list)  # [(event_name, html_payload), ...]
    output_item: dict = field(default_factory=dict)                   # payload for conversations.items.create
