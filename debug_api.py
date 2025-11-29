import requests
import json

API_URL = "http://localhost:8000"

def test_endpoints():
    print("Testing API endpoints...")
    
    # Test /model-info
    try:
        print("\n1. Testing /model-info...")
        response = requests.get(f"{API_URL}/model-info")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:", json.dumps(response.json(), indent=2)[:200] + "...")
        else:
            print("Error:", response.text)
    except Exception as e:
        print(f"Exception: {e}")

    # Test /example-results
    try:
        print("\n2. Testing /example-results...")
        response = requests.get(f"{API_URL}/example-results")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:", json.dumps(response.json(), indent=2)[:200] + "...")
        else:
            print("Error:", response.text)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_endpoints()
