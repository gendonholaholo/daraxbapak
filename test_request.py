import requests
import json

url = "http://localhost:8000/api/v1/interview"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}
data = {
    "session_id": "test-session-1",
    "message": "Ceritakan tentang pengalaman kerja Anda",
    "topic": "pengalaman kerja",
    "question": "Ceritakan tentang pengalaman kerja Anda?",
    "point": None
}

response = requests.post(url, headers=headers, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
print(f"Response Body: {json.dumps(response.json(), indent=2)}") 