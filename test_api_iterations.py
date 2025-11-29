import requests
import base64
import io
from PIL import Image
import numpy as np

API_URL = "http://localhost:8000"

def create_dummy_image():
    """Create a dummy RGB image"""
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_attack_iterations():
    print("Testing /attack-iterations endpoint...")
    
    # Create dummy image
    img_file = create_dummy_image()
    
    files = {'file': ('test.png', img_file, 'image/png')}
    data = {
        'epsilon': 0.03,
        'pgd_alpha': 0.0075,
        'pgd_iterations': 5,
        'random_start': True
    }
    
    try:
        response = requests.post(f"{API_URL}/attack-iterations", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("[OK] Request successful")
            print(f"Original Class: {result['original_class']}")
            print(f"Total Iterations: {result['total_iterations']}")
            print(f"Final Success: {result['final_success']}")
            
            # Verify iterations structure
            iterations = result['iterations']
            if len(iterations) == 5:
                print(f"[OK] Returned correct number of iterations: {len(iterations)}")
            else:
                print(f"[FAIL] Expected 5 iterations, got {len(iterations)}")
                
            # Check first iteration structure
            first_iter = iterations[0]
            required_keys = ['iteration', 'image', 'predicted_class', 'confidence', 'is_adversarial', 'perturbation_linf']
            if all(key in first_iter for key in required_keys):
                print("[OK] Iteration data structure is correct")
            else:
                print(f"[FAIL] Missing keys in iteration data. Found: {first_iter.keys()}")
                
        else:
            print(f"[FAIL] Request failed with status code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"[FAIL] Error during request: {str(e)}")

if __name__ == "__main__":
    test_attack_iterations()
