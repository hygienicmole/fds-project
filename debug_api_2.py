import requests
import json

API_URL = "http://localhost:8000"

def test_example_results():
    print("Testing /example-results...")
    try:
        response = requests.get(f"{API_URL}/example-results")
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print("Error Response:")
            print(response.text)
        else:
            print("Success")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_example_results()
