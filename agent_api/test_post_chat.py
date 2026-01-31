from fastapi.testclient import TestClient
import importlib

app = importlib.import_module('app.main').app
client = TestClient(app)

resp = client.post('/chat', json={'message':'List some portfolio you have','conversation_history':[]})
print('Status:', resp.status_code)
print('Text:', resp.text)
print('JSON:', None)
try:
    print('Parsed JSON:', resp.json())
except Exception as e:
    print('JSON parse error:', e)
    import traceback
    traceback.print_exc()
