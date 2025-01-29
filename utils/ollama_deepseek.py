import ollama

MODEL = "deepseek-r1:8b"
URL = "http://localhost:11434/api/generate"

# def ask_question(query):
        
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "deepseek-r1:8b",
#             "prompt": query,
#             "stream": False
#         }
#     )
#     return response.json()["response"]


res = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "why is the ocean so salty"
        }
    ],
    stream=True,
)
for chunk in res:
    print(chunk["message"]["content"], end="", flush=True)


