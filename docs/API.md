# Adversarial ML API Documentation

Complete API reference for the FastAPI backend that provides adversarial attack generation and analysis capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [API Endpoints](#api-endpoints)
4. [Request/Response Schemas](#request-response-schemas)
5. [Usage Examples](#usage-examples)
6. [Error Handling](#error-handling)
7. [Deployment](#deployment)
8. [Frontend Integration](#frontend-integration)

---

## Overview

The Adversarial ML API provides a RESTful interface for:
- Uploading and classifying images
- Generating adversarial examples using FGSM and PGD attacks
- Retrieving model information and statistics
- Accessing pre-computed example results

**Base URL**: `http://localhost:8000`

**Interactive Documentation**: `http://localhost:8000/docs`

**Alternative Documentation**: `http://localhost:8000/redoc`

---

## Getting Started

### Installation

1. **Install dependencies**:
```bash
pip install fastapi uvicorn python-multipart pillow
```

Or install all backend requirements:
```bash
pip install -r requirements.txt
```

2. **Ensure model is trained**:
```bash
python train_baseline.py
```
The API expects a trained model at `models/baseline_best_model.pth`.

### Running the Server

**Development mode** (with auto-reload):
```bash
uvicorn app.backend:app --reload --host 0.0.0.0 --port 8000
```

**Production mode**:
```bash
uvicorn app.backend:app --host 0.0.0.0 --port 8000 --workers 4
```

**Using Python directly**:
```bash
python -m app.backend
```

### Verify Server is Running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-16T00:00:00",
  "model_loaded": true,
  "device": "cuda"
}
```

---

## API Endpoints

### 1. Root Endpoint

**GET /**

Get API information and available endpoints.

**Response**:
```json
{
  "name": "Adversarial ML API",
  "version": "1.0.0",
  "description": "REST API for adversarial example generation",
  "endpoints": { ... },
  "documentation": { ... }
}
```

---

### 2. Model Information

**GET /model-info**

Get information about the loaded model.

**Response**:
```json
{
  "model_name": "ResNet18",
  "num_classes": 10,
  "class_names": ["airplane", "automobile", "bird", ...],
  "parameters": 11173962,
  "device": "cuda",
  "model_loaded": true,
  "checkpoint_info": {
    "epoch": 200,
    "accuracy": 91.45,
    "loss": 0.234
  }
}
```

**Example**:
```bash
curl http://localhost:8000/model-info
```

---

### 3. Upload Image

**POST /upload**

Upload an image and get model predictions.

**Parameters**:
- `file` (form-data, required): Image file (PNG, JPG, JPEG)

**Response**:
```json
{
  "filename": "cat.jpg",
  "size": [800, 600],
  "prediction": {
    "predicted_class": 3,
    "predicted_label": "cat",
    "confidence": 0.92,
    "all_probabilities": {
      "airplane": 0.01,
      "automobile": 0.02,
      "bird": 0.03,
      "cat": 0.92,
      ...
    }
  },
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "message": "Image uploaded and classified successfully"
}
```

**Example**:
```bash
curl -X POST \
  -F "file=@cat_image.jpg" \
  http://localhost:8000/upload
```

**Python Example**:
```python
import requests

url = "http://localhost:8000/upload"
files = {"file": open("cat.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

### 4. Generate Adversarial Attack

**POST /attack**

Generate an adversarial example for an uploaded image.

**Parameters** (form-data):
- `file` (required): Image file to attack
- `attack_type` (optional): Attack type - "fgsm" or "pgd" (default: "fgsm")
- `epsilon` (optional): Perturbation magnitude 0.0-1.0 (default: 0.03)
- `targeted` (optional): Perform targeted attack (default: false)
- `target_class` (optional): Target class 0-9 (required if targeted=true)
- `pgd_alpha` (optional): PGD step size (default: 0.01)
- `pgd_iterations` (optional): PGD iterations (default: 20)
- `random_start` (optional): Use random start for PGD (default: true)

**Response**:
```json
{
  "success": true,
  "original_prediction": {
    "predicted_class": 3,
    "predicted_label": "cat",
    "confidence": 0.92,
    "all_probabilities": { ... }
  },
  "adversarial_prediction": {
    "predicted_class": 5,
    "predicted_label": "dog",
    "confidence": 0.67,
    "all_probabilities": { ... }
  },
  "original_image": "data:image/png;base64,...",
  "adversarial_image": "data:image/png;base64,...",
  "perturbation": "data:image/png;base64,...",
  "attack_parameters": {
    "attack_type": "pgd",
    "epsilon": 0.03,
    "targeted": false,
    "target_class": null,
    "pgd_alpha": 0.01,
    "pgd_iterations": 20,
    "random_start": true
  },
  "perturbation_stats": {
    "l0_norm": 3052.0,
    "l1_norm": 91.2,
    "l2_norm": 1.66,
    "linf_norm": 0.03,
    "mean_perturbation": 0.0297,
    "max_perturbation": 0.03
  }
}
```

**Examples**:

**FGSM Attack**:
```bash
curl -X POST \
  -F "file=@cat.jpg" \
  -F "attack_type=fgsm" \
  -F "epsilon=0.03" \
  http://localhost:8000/attack
```

**PGD Attack**:
```bash
curl -X POST \
  -F "file=@cat.jpg" \
  -F "attack_type=pgd" \
  -F "epsilon=0.03" \
  -F "pgd_iterations=40" \
  http://localhost:8000/attack
```

**Targeted Attack**:
```bash
curl -X POST \
  -F "file=@cat.jpg" \
  -F "attack_type=pgd" \
  -F "epsilon=0.05" \
  -F "targeted=true" \
  -F "target_class=5" \
  http://localhost:8000/attack
```

**Python Example**:
```python
import requests

url = "http://localhost:8000/attack"
files = {"file": open("cat.jpg", "rb")}
data = {
    "attack_type": "pgd",
    "epsilon": 0.03,
    "pgd_iterations": 20
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Attack Success: {result['success']}")
print(f"Original: {result['original_prediction']['predicted_label']}")
print(f"Adversarial: {result['adversarial_prediction']['predicted_label']}")
```

---

### 5. Example Results

**GET /example-results**

Get pre-computed example attack results from evaluation experiments.

**Response**:
```json
[
  {
    "image_id": 1,
    "attack_type": "FGSM",
    "epsilon": 0.03,
    "original_class": "cat",
    "adversarial_class": "dog",
    "attack_success": true,
    "original_confidence": 0.92,
    "adversarial_confidence": 0.67
  },
  {
    "image_id": 2,
    "attack_type": "PGD-20",
    "epsilon": 0.03,
    "original_class": "airplane",
    "adversarial_class": "bird",
    "attack_success": true,
    "original_confidence": 0.95,
    "adversarial_confidence": 0.78
  },
  ...
]
```

**Example**:
```bash
curl http://localhost:8000/example-results
```

---

### 6. Health Check

**GET /health**

Check API health status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-16T00:00:00.123456",
  "model_loaded": true,
  "device": "cuda"
}
```

**Example**:
```bash
curl http://localhost:8000/health
```

---

## Request/Response Schemas

### PredictionResponse

```typescript
{
  predicted_class: number,      // 0-9
  predicted_label: string,       // "cat", "dog", etc.
  confidence: number,            // 0.0-1.0
  all_probabilities: {
    [class_name: string]: number // 0.0-1.0
  }
}
```

### AttackResponse

```typescript
{
  success: boolean,
  original_prediction: PredictionResponse,
  adversarial_prediction: PredictionResponse,
  original_image: string,        // Base64-encoded PNG
  adversarial_image: string,     // Base64-encoded PNG
  perturbation: string,          // Base64-encoded PNG
  attack_parameters: {
    attack_type: "fgsm" | "pgd",
    epsilon: number,
    targeted: boolean,
    target_class: number | null,
    pgd_alpha?: number,
    pgd_iterations?: number,
    random_start?: boolean
  },
  perturbation_stats: {
    l0_norm: number,
    l1_norm: number,
    l2_norm: number,
    linf_norm: number,
    mean_perturbation: number,
    max_perturbation: number
  }
}
```

---

## Usage Examples

### Complete Python Client Example

```python
import requests
import base64
from PIL import Image
import io

class AdversarialMLClient:
    """Client for Adversarial ML API"""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def get_model_info(self):
        """Get model information"""
        response = requests.get(f"{self.base_url}/model-info")
        return response.json()

    def upload_image(self, image_path):
        """Upload and classify an image"""
        files = {"file": open(image_path, "rb")}
        response = requests.post(f"{self.base_url}/upload", files=files)
        return response.json()

    def generate_attack(self, image_path, attack_type="fgsm", epsilon=0.03, **kwargs):
        """Generate adversarial example"""
        files = {"file": open(image_path, "rb")}
        data = {
            "attack_type": attack_type,
            "epsilon": epsilon,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/attack", files=files, data=data)
        return response.json()

    def decode_base64_image(self, base64_string):
        """Decode base64 image to PIL Image"""
        # Remove data URL prefix if present
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))

    def get_examples(self):
        """Get pre-computed examples"""
        response = requests.get(f"{self.base_url}/example-results")
        return response.json()


# Usage
client = AdversarialMLClient()

# Get model info
info = client.get_model_info()
print(f"Model: {info['model_name']}, Accuracy: {info['checkpoint_info']['accuracy']}%")

# Upload image
result = client.upload_image("cat.jpg")
print(f"Predicted: {result['prediction']['predicted_label']} ({result['prediction']['confidence']:.2%})")

# Generate FGSM attack
attack_result = client.generate_attack("cat.jpg", attack_type="fgsm", epsilon=0.03)
print(f"Attack Success: {attack_result['success']}")
print(f"Original: {attack_result['original_prediction']['predicted_label']}")
print(f"Adversarial: {attack_result['adversarial_prediction']['predicted_label']}")

# Save adversarial image
adv_image = client.decode_base64_image(attack_result['adversarial_image'])
adv_image.save("adversarial_cat.png")

# Generate PGD attack
pgd_result = client.generate_attack(
    "cat.jpg",
    attack_type="pgd",
    epsilon=0.03,
    pgd_iterations=40
)
```

### JavaScript/TypeScript Example

```typescript
// Using Fetch API
async function uploadImage(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/upload', {
    method: 'POST',
    body: formData
  });

  return await response.json();
}

async function generateAttack(file: File, params: {
  attack_type?: string,
  epsilon?: number,
  pgd_iterations?: number
}) {
  const formData = new FormData();
  formData.append('file', file);

  Object.entries(params).forEach(([key, value]) => {
    formData.append(key, String(value));
  });

  const response = await fetch('http://localhost:8000/attack', {
    method: 'POST',
    body: formData
  });

  return await response.json();
}

// Usage in React component
function AttackDemo() {
  const [result, setResult] = useState(null);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const attackResult = await generateAttack(file, {
      attack_type: 'pgd',
      epsilon: 0.03,
      pgd_iterations: 20
    });

    setResult(attackResult);
  };

  return (
    <div>
      <input type="file" onChange={handleFileSelect} />
      {result && (
        <div>
          <img src={result.original_image} alt="Original" />
          <img src={result.adversarial_image} alt="Adversarial" />
          <p>Success: {result.success ? 'Yes' : 'No'}</p>
        </div>
      )}
    </div>
  );
}
```

---

## Error Handling

The API uses standard HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **422**: Validation Error (schema mismatch)
- **500**: Internal Server Error
- **503**: Service Unavailable (model not loaded)

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

**Model not loaded**:
```json
{
  "detail": "Model not loaded. Please check server logs."
}
```
**Solution**: Ensure model checkpoint exists at `models/baseline_best_model.pth`

**Invalid epsilon**:
```json
{
  "detail": "epsilon must be between 0 and 1"
}
```
**Solution**: Use epsilon in range [0.0, 1.0]

**Invalid attack type**:
```json
{
  "detail": "attack_type must be 'fgsm' or 'pgd'"
}
```
**Solution**: Use "fgsm" or "pgd"

**Targeted attack without target**:
```json
{
  "detail": "target_class must be specified (0-9) for targeted attacks"
}
```
**Solution**: Provide target_class when targeted=true

---

## Deployment

### Local Development

```bash
uvicorn app.backend:app --reload --host 127.0.0.1 --port 8000
```

### Production Deployment

**Using Gunicorn with Uvicorn workers**:
```bash
gunicorn app.backend:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

**Using Docker**:

Create `Dockerfile`:
```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t adversarial-ml-api .
docker run -p 8000:8000 adversarial-ml-api
```

### Environment Variables

```bash
# Model path
export MODEL_PATH="models/baseline_best_model.pth"

# Device
export DEVICE="cuda"  # or "cpu"

# API host and port
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

---

## Frontend Integration

### CORS Configuration

The API is configured with permissive CORS settings for development:
- `allow_origins=["*"]` - All origins allowed
- `allow_methods=["*"]` - All HTTP methods allowed
- `allow_headers=["*"]` - All headers allowed

**Production**: Update CORS settings in `app/backend.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### React Integration Example

```jsx
import React, { useState } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

function AdversarialAttackUI() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAttack = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('attack_type', 'pgd');
    formData.append('epsilon', '0.03');

    try {
      const response = await axios.post(`${API_URL}/attack`, formData);
      setResult(response.data);
    } catch (error) {
      console.error('Attack failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={handleAttack} disabled={!file || loading}>
        {loading ? 'Generating...' : 'Generate Attack'}
      </button>

      {result && (
        <div>
          <h3>Results</h3>
          <div style={{ display: 'flex', gap: '20px' }}>
            <div>
              <h4>Original</h4>
              <img src={result.original_image} alt="Original" />
              <p>{result.original_prediction.predicted_label}</p>
              <p>{(result.original_prediction.confidence * 100).toFixed(2)}%</p>
            </div>
            <div>
              <h4>Adversarial</h4>
              <img src={result.adversarial_image} alt="Adversarial" />
              <p>{result.adversarial_prediction.predicted_label}</p>
              <p>{(result.adversarial_prediction.confidence * 100).toFixed(2)}%</p>
            </div>
            <div>
              <h4>Perturbation</h4>
              <img src={result.perturbation} alt="Perturbation" />
            </div>
          </div>
          <p>Attack Success: {result.success ? 'Yes' : 'No'}</p>
        </div>
      )}
    </div>
  );
}
```

---

## Performance Considerations

### Response Times

- **GET /model-info**: ~10ms
- **POST /upload**: ~50-100ms (CPU), ~20-50ms (GPU)
- **POST /attack** (FGSM): ~100-200ms (CPU), ~50-100ms (GPU)
- **POST /attack** (PGD-20): ~2-4s (CPU), ~500ms-1s (GPU)
- **GET /example-results**: ~5-10ms

### Optimization Tips

1. **Use GPU**: Ensure CUDA is available for faster inference
2. **Reduce PGD iterations**: Use 10-20 iterations for interactive applications
3. **Batch processing**: Process multiple images in parallel
4. **Caching**: Cache model predictions for repeated images
5. **Async processing**: Use background tasks for long-running attacks

### Scaling

For high-traffic scenarios:
- Use load balancer (nginx, HAProxy)
- Deploy multiple worker processes
- Use Redis for caching
- Consider GPU cluster for parallel processing

---

## Security Considerations

### Input Validation

- File size limits enforced
- Image format validation
- Parameter range checking
- Schema validation with Pydantic

### Rate Limiting

Add rate limiting for production:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/attack")
@limiter.limit("10/minute")
async def generate_attack(...):
    ...
```

### Authentication

For production, add authentication:
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/attack")
async def generate_attack(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    ...
):
    # Validate token
    ...
```

---

## Troubleshooting

### Model Not Loading

**Issue**: `Model not loaded` error

**Solutions**:
1. Check model path: `models/baseline_best_model.pth` exists
2. Train model: `python train_baseline.py`
3. Check permissions on model file
4. Review server logs for detailed error

### CUDA Out of Memory

**Issue**: GPU memory errors

**Solutions**:
1. Use CPU: Set `DEVICE='cpu'` in backend.py
2. Reduce batch size (affects future batch processing)
3. Clear GPU cache: `torch.cuda.empty_cache()`

### Slow Response Times

**Issue**: Requests taking too long

**Solutions**:
1. Use GPU instead of CPU
2. Reduce PGD iterations
3. Deploy with more workers
4. Use async processing for heavy computations

---

## API Changelog

### Version 1.0.0 (2025-11-16)

- Initial release
- FGSM and PGD attack support
- Image upload and classification
- Model information endpoint
- Pre-computed examples endpoint
- CORS support
- Comprehensive error handling
- Interactive API documentation

---

## Contact and Support

For issues, questions, or contributions:
- GitHub Issues: Create an issue in the repository
- Documentation: Check [docs/methodology.md](methodology.md) for attack details
- Examples: See [docs/experiments.md](experiments.md) for experimental results

---

**API Version**: 1.0.0
**Last Updated**: 2025-11-16
**Documentation**: http://localhost:8000/docs
