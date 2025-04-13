from fastapi import Depends
from openai import AsyncOpenAI


# Helper function to get or create a vector store
async def get_or_create_vector_store(assistantId: str, client: AsyncOpenAI = Depends(lambda: AsyncOpenAI())) -> str:
    assistant = await client.beta.assistants.retrieve(assistantId)
    if assistant.tool_resources and assistant.tool_resources.file_search and assistant.tool_resources.file_search.vector_store_ids:
        return assistant.tool_resources.file_search.vector_store_ids[0]
    vector_store = await client.vector_stores.create(name="sample-assistant-vector-store") # TODO: Make this dynamic
    await client.beta.assistants.update(
        assistantId,
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store.id],
            },
        }
    )
    return vector_store.id