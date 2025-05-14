from fastapi.testclient import TestClient
from solution_guidance.app import app

client = TestClient(app)

def test_train_endpoint():
    response = client.post('/train')
    assert response.status_code == 200

# Additional test cases for prediction and monitoring