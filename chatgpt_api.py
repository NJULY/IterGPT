import json
import requests

URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "put your API KEY here"  # API KEY is like "sk-xxxxxxxxxxxxxx"
}


RETRY_NUM = 10

def chatgpt(message, model="gpt-4o-mini"):
    data = {
        "model": model,
        "temperature": 0.,
        "presence_penalty": 0.,
        "frequency_penalty": 0.,
        "top_p": 1,
        "messages": message,
    }

    retry_cnt = 0
    while True:
        if retry_cnt <= RETRY_NUM:
            try:
                response = requests.post(URL, headers=HEADERS, data=json.dumps(data).encode('utf-8'))
                response = json.loads(response.content.decode("utf-8"))
                break
            except Exception as ex:
                retry_cnt += 1
        else:
            exit(0)

    response_text = response["choices"][0]["message"]["content"]
    return response_text