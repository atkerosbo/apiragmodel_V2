from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = "gpt-4o-mini"
KEY = os.getenv("OPEN_AI_KEY")
ORG = os.getenv("ORG")
PROJECT = os.getenv("PROJECT")
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_KEY")

client = OpenAI()



def extract_chat_message(response):
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError):
        return None

def chat_prompt_openai(query):
    completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": query}
             ]
            )
    message = extract_chat_message(completion)
    return message


def chat_with_context(conversation):
    completion = client.chat.completions.create(
    model=MODEL,
    messages=conversation
            )
    message = extract_chat_message(completion)
    return message


########### AI PROMPTS WITH TEMPLATES ############

def get_keywords_with_openai(query: str):
    # print(query)
    query_keywords = (f"Get me the keywords for an SQL query from this question: {query}. Just return a simple list of keywords divided by commas.")
    response = chat_prompt_openai(query_keywords)
    #print(response)
    return response

def answer_tamplate(query: str, context: str):
    response = (f"Format these query results to resembles a list a search rusults for this query answered by a sales assistent: {query}. This was the users query: {context}")
    answer = chat_prompt_openai(response)
    return answer

def get_type_of_query(query: str):
    query_type = (f" You are a helpfull assistant that categorizes a sentence. You are given a sentence and you need to categorize which category it belongs to. Is it Product related, Or general information related.Output only the category, Product or General this is the sentence: {query}")
    answer = chat_prompt_openai(query_type)
    #print(answer)
    return answer

def complete_answer_with_context(context: str):
    complete_answer = (f"Format this list of products {context} in serbian language so i can display them as a search result ")
    answer = chat_prompt_openai(complete_answer)
    return answer

def summerize_answer(query, chunk_to_summerize):
    response_to_summerize = (f"Skrati mi tekst u nekoliko recenica i izvuci sustinu na osnovu ovog pitanja {query} .Ceo tekst :{chunk_to_summerize}")
    summerized_response = chat_prompt_openai(response_to_summerize)
    return summerized_response


# messages = [
#   {"role": "system", "content": "You are a helpful assistant."},
#   {"role": "user","content": "remenik za kombajn"}
# ]

messages2 = [
  {
    "role": "user",
    "content": "remenik za kombajn"
  },
  {
    "role": "system",
    "answer": [
      {
        "code": "1006319",
        "name": "3HB2700 La \"DUNLOP\"",
        "product URL": "https://www.agrimar.rs/3hb2700-la-dunlop_1006319/",
        "description": "SPOJENI  KLINASTI  REMEN",
        "price": "3422.22 RSD"
      },
      {
        "code": "1006334",
        "name": "3HC2730 La \"DUNLOP\"",
        "product URL": "https://www.agrimar.rs/3hc2730-la-dunlop_1006334/",
        "description": "SPOJENI  KLINASTI  REMEN",
        "price": "5697.31 RSD"
      },
      {
        "code": "1006336",
        "name": "3HC4200 La \"DUNLOP\"",
        "product URL": "https://www.agrimar.rs/3hc4200-la-dunlop_1006336/",
        "description": "SPOJENI  KLINASTI  REMEN",
        "price": "8765.09 RSD"
      },
      {
        "code": "1006987",
        "name": "2HB2440 La \"DUNLOP\" (CL 667251.1)",
        "product URL": "https://www.agrimar.rs/2hb2440-la-dunlop-cl-6672511_1006987/",
        "description": "SPOJENI  KLINASTI REMEN",
        "price": "1996.21 RSD"
      },
      {
        "code": "1011345",
        "name": "zakovica JUS MB3 011 6x16",
        "product URL": "https://www.agrimar.rs/zakovica-jus-mb3-011-6x16_1011345/",
        "description": "ZA KOMBAJN",
        "price": "2.84 RSD"
      }
    ]
  }
]

answer = chat_with_context(messages2)
print(answer)