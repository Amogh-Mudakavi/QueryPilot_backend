from llm.router import router
from llm.providers import LLMError


def generate_title(message, default="New Conversation"):
    try:
        response = router.generate(
            task="title",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a short conversation title "
                        "in maximum 5 words. "
                        "Return only the title."
                    )
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except LLMError:
        return default
    
    