from dotenv import load_dotenv
import os
from datetime import datetime
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

@tool
def get_string_length(text:str)->int:
    """return the exact number of character in a given string.
    use this when the user asks for the length or character count of a text.
    """
    return len(text)


@tool
def analyze_message_tone(message:str)->str:
    """analyze the emotional tone of a user's message.
    Return exactly one of the these cotegories: 'funny','serious','angry',anxious'
    """
    msg_lower = message.lower()
    if any(w in msg_lower for w in ["anger","angry","mad","hate"]):return "angry"
    if any(w in msg_lower for w in ["serious","urgent","deadline","asap"]):return "serious"
    if any(w in msg_lower for w in ["comedy","laugh","joke","fun","funny"]):return "funny"
    if any(w in msg_lower for w in ["anxiety","anxious","afraid","nervous"]):return "anxious"
    return "neutral"

@tool
def get_current_time(text:str)->str:
    """returns current date and time.use this only when explicilty ask what time it is."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_text_statictics(text:str)->dict:
   """calculate words character and sentence count for a given text."""
   return {
       "word_count":len(text.split()),
       "character_count":len(text),
       "sentence_count":text.count('.') + text.count('!') + text.count('?')
   }

tools=[
    get_text_statictics,
    get_current_time,
    analyze_message_tone,
    get_string_length
]

llm=ChatMistralAI(model_name="mistral-small-2603",temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system","You are an helpful ai assistant. use the provided tools to answer the user's question accurately.if a tool returns an error explain it to the user"),
    ("human","{input}"),
    
])

agent = create_agent(llm,tools,prompt)


if __name__ == "__main__":
    print("Agent started. Type 'exit' to quit.")

    while True:
        query = input("\nYou: ")

        if query.lower() == "exit":
            break

        try:
            result = agent.invoke({
                "input": query
            })

            print("\nAssistant:", result["output"])

        except Exception as e:
            print(f"Error: {e}")