import speech_recognition as sr
from graph import graph
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import asyncio


load_dotenv()

messages = []

openai = AsyncOpenAI()

async def tts(text: str):
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions="Speak in cheerful and positive tone.",
        response_format="pcm"
    ) as response:
        await LocalAudioPlayer().play(response)

def main():
    r = sr.Recognizer()
    
    # Access microphone
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)  # Reduce noise

        r.pause_threshold = 2 # If user doesnt speak for 2 sec, stop listening start further process

        while True: 
            print("Speak something...")
            audio = r.listen(source)
            
            print("Processing audio... (STT)")
            stt = r.recognize_google(audio)

            print("You said ..", stt)
            messages.append({"role" : "user", "content" : stt})

            for event in graph.stream({"messages" : messages}, stream_mode="values"):
                if "messages" in event:
                    messages.append({"role" : "assistant", "content" : event["messages"][-1].content})
                    event["messages"][-1].pretty_print()



# main()

asyncio.run(tts(text="Hey Nice to meet you how are you "))