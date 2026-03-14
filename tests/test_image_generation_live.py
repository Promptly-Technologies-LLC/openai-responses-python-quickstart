"""
Live integration tests for image generation tool support.

These tests hit the real OpenAI API to observe:
1. The image_generation tool payload is accepted
2. The streaming event sequence and types
3. Whether text deltas are sent alongside image generation calls

Requires a valid OPENAI_API_KEY in .env or environment.
Mark: all tests use @pytest.mark.live to allow selective runs.
"""

import pytest
from openai import AsyncOpenAI

from conftest import REAL_API_KEY

_REAL_API_KEY = REAL_API_KEY
_has_real_key = bool(_REAL_API_KEY)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _has_real_key, reason="No real OPENAI_API_KEY available"),
]

MODEL = "gpt-4.1-mini"

IG_TOOL = {"type": "image_generation"}


@pytest.fixture
async def client():
    c = AsyncOpenAI(api_key=_REAL_API_KEY)
    yield c
    await c.close()


class TestImageGenerationEventSequence:
    """Observe the real event sequence from an image generation call."""

    @pytest.mark.anyio
    async def test_image_generation_event_types(self, client: AsyncOpenAI):
        """Log all event types from an image generation request."""
        stream = await client.responses.create(
            model=MODEL,
            tools=[IG_TOOL],
            input="Generate an image of a small red circle on a white background.",
            stream=True,
        )

        events = []
        async with stream as event_stream:
            async for event in event_stream:
                name = type(event).__name__
                events.append((name, event))

        event_types = [e[0] for e in events]

        # Print all event types for observation
        print("\n=== Image Generation Event Types ===")
        for i, (name, event) in enumerate(events):
            extra = ""
            if hasattr(event, "item_id"):
                extra += f" item_id={event.item_id}"
            if hasattr(event, "output_index"):
                extra += f" output_index={event.output_index}"
            if hasattr(event, "type") and isinstance(event.type, str):
                extra += f" type={event.type}"
            # For output item events, show the item type
            if hasattr(event, "item") and hasattr(event.item, "type"):
                extra += f" item.type={event.item.type}"
            print(f"  [{i:3d}] {name}{extra}")

        # Check for text deltas
        has_text_deltas = "ResponseTextDeltaEvent" in event_types
        print(f"\n=== Text deltas present: {has_text_deltas} ===")
        if has_text_deltas:
            text_events = [(i, e) for i, (n, e) in enumerate(events)
                           if n == "ResponseTextDeltaEvent"]
            print(f"  Count: {len(text_events)}")
            sample_text = "".join(
                e.delta for _, e in text_events[:20]
                if hasattr(e, "delta")
            )
            print(f"  Sample text: {sample_text[:200]}")

        # Check for image generation events
        ig_event_names = [n for n in event_types if "ImageGen" in n]
        print(f"\n=== Image generation events: {ig_event_names} ===")

        # Stream must complete
        assert "ResponseCompletedEvent" in event_types

    @pytest.mark.anyio
    async def test_image_generation_produces_image_events(self, client: AsyncOpenAI):
        """An image generation request should produce image-specific events."""
        stream = await client.responses.create(
            model=MODEL,
            tools=[IG_TOOL],
            input="Draw a simple blue square.",
            stream=True,
        )

        event_types = []
        async with stream as event_stream:
            async for event in event_stream:
                event_types.append(type(event).__name__)

        # Should have at least some image generation events
        ig_events = [e for e in event_types if "ImageGen" in e]
        assert len(ig_events) >= 1, (
            f"Expected image generation events, got: {event_types}"
        )

        assert "ResponseCompletedEvent" in event_types

    @pytest.mark.anyio
    async def test_text_deltas_alongside_image_generation(self, client: AsyncOpenAI):
        """Check whether the model sends text alongside image generation."""
        stream = await client.responses.create(
            model=MODEL,
            tools=[IG_TOOL],
            input="Generate an image of a green triangle and describe what you created.",
            stream=True,
        )

        event_types = []
        text_content = []
        async with stream as event_stream:
            async for event in event_stream:
                name = type(event).__name__
                event_types.append(name)
                if name == "ResponseTextDeltaEvent" and hasattr(event, "delta"):
                    text_content.append(event.delta)

        has_text = "ResponseTextDeltaEvent" in event_types
        has_image = any("ImageGen" in e for e in event_types)

        print("\n=== Results ===")
        print(f"  Has text deltas: {has_text}")
        print(f"  Has image gen events: {has_image}")
        if text_content:
            print(f"  Text: {''.join(text_content)[:300]}")

        # This test is observational — it always passes but logs the behavior
        assert "ResponseCompletedEvent" in event_types
