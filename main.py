from dotenv import load_dotenv
from datetime import datetime

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()


@tool
def get_string_length(text: str) -> int:
    """
    Return the exact number of characters in a string.
    Use when the user asks for character count or string length.
    """
    return len(text)


@tool
def analyze_message_tone(message: str) -> str:
    """
    Analyze the emotional tone of a message.
    Return one of:
    funny, serious, angry, anxious, neutral
    """
    msg_lower = message.lower()

    if any(word in msg_lower for word in ["anger", "angry", "mad", "hate"]):
        return "angry"

    if any(word in msg_lower for word in ["serious", "urgent", "deadline", "asap"]):
        return "serious"

    if any(word in msg_lower for word in ["comedy", "laugh", "joke", "fun", "funny"]):
        return "funny"

    if any(word in msg_lower for word in ["anxiety", "anxious", "afraid", "nervous"]):
        return "anxious"

    return "neutral"


@tool
def get_current_time(_: str = "") -> str:
    """
    Return the current date and time.
    Use only when the user explicitly asks for the time.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_text_statistics(text: str) -> dict:
    """
    Calculate word, character, and sentence counts.
    """
    return {
        "word_count": len(text.split()),
        "character_count": len(text),
        "sentence_count": text.count(".") + text.count("!") + text.count("?"),
    }


tools = [
    get_string_length,
    analyze_message_tone,
    get_current_time,
    get_text_statistics,
]


llm = ChatMistralAI(
    model_name="mistral-small-2603",
    temperature=0.3,
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
    You are a helpful AI assistant.
    Use tools whenever appropriate.
    If a tool fails, explain the error to the user.
    """,
)


if __name__ == "__main__":
    print("Agent started. Type 'exit' to quit.")

    while True:
        query = input("\nYou: ")

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query,
                        }
                    ]
                }
            )


            # Usually the final response is the last message
            if "messages" in result:
                print(
                    "\nAssistant:",
                    result["messages"][-1].content
                )

        except Exception as e:
            print(f"\nError: {e}")