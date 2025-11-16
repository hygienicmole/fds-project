"""
FastAPI Backend for Adversarial ML Project

This module provides a RESTful API for generating and analyzing adversarial examples.
It exposes endpoints for image upload, adversarial attack generation, model information,
and pre-computed example results.

The backend supports:
- FGSM and PGD attacks with configurable parameters
- Real-time adversarial example generation
- Confidence score analysis
- Model performance statistics
- CORS for frontend integration

Example:
    Start the server:
    ```bash
    uvicorn app.backend:app --reload --host 0.0.0.0 --port 8000
    ```

    Test endpoints:
    ```bash
    curl http://localhost:8000/model-info
    curl -X POST -F "file=@image.png" http://localhost:8000/upload
    ```

Requirements:
    - FastAPI
    - uvicorn
    - python-multipart (for file uploads)
    - Pillow (for image processing)
    - PyTorch
    - numpy

Install:
    ```bash
    pip install fastapi uvicorn python-multipart pillow torch numpy
    ```
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
import os
import sys
import json
from datetime import datetime
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_resnet18, load_checkpoint
from attacks import FGSM, PGD
import config

# ============================================================================
# Configuration and Constants
# ============================================================================

# Model checkpoint path
MODEL_PATH = os.path.join('models', 'baseline_best_model.pth')

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Normalization parameters (CIFAR-10)
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2470, 0.2435, 0.2616]

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class AttackRequest(BaseModel):
    """
    Request schema for adversarial attack generation.

    Attributes:
        attack_type: Type of attack ('fgsm' or 'pgd')
        epsilon: Perturbation magnitude (0.0-1.0)
        targeted: Whether to perform targeted attack
        target_class: Target class index for targeted attacks
        pgd_alpha: Step size for PGD (only for PGD attack)
        pgd_iterations: Number of iterations for PGD
        random_start: Use random initialization for PGD
    """
    attack_type: Literal['fgsm', 'pgd'] = Field(
        default='fgsm',
        description="Type of adversarial attack"
    )
    epsilon: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Perturbation magnitude (L-infinity norm)"
    )
    targeted: bool = Field(
        default=False,
        description="Perform targeted attack"
    )
    target_class: Optional[int] = Field(
        default=None,
        ge=0,
        le=9,
        description="Target class for targeted attack (0-9)"
    )
    pgd_alpha: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Step size for PGD attack"
    )
    pgd_iterations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of PGD iterations"
    )
    random_start: bool = Field(
        default=True,
        description="Use random initialization for PGD"
    )

    @validator('target_class')
    def validate_target_class(cls, v, values):
        """Validate that target_class is provided for targeted attacks."""
        if values.get('targeted') and v is None:
            raise ValueError('target_class must be specified for targeted attacks')
        return v


class PredictionResponse(BaseModel):
    """
    Response schema for model predictions.

    Attributes:
        predicted_class: Index of predicted class
        predicted_label: Name of predicted class
        confidence: Confidence score (0-1)
        all_probabilities: Probabilities for all classes
    """
    predicted_class: int
    predicted_label: str
    confidence: float
    all_probabilities: Dict[str, float]


class AttackResponse(BaseModel):
    """
    Response schema for adversarial attack results.

    Attributes:
        success: Whether the attack was successful
        original_prediction: Prediction on original image
        adversarial_prediction: Prediction on adversarial image
        original_image: Base64-encoded original image
        adversarial_image: Base64-encoded adversarial image
        perturbation: Base64-encoded perturbation visualization
        attack_parameters: Parameters used for the attack
        perturbation_stats: Statistics about the perturbation
    """
    success: bool
    original_prediction: PredictionResponse
    adversarial_prediction: PredictionResponse
    original_image: str
    adversarial_image: str
    perturbation: str
    attack_parameters: Dict[str, Any]
    perturbation_stats: Dict[str, float]


class ModelInfoResponse(BaseModel):
    """
    Response schema for model information.

    Attributes:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        class_names: List of class names
        parameters: Number of model parameters
        device: Device model is loaded on
        model_loaded: Whether model is successfully loaded
        checkpoint_info: Information from model checkpoint
    """
    model_name: str
    num_classes: int
    class_names: List[str]
    parameters: int
    device: str
    model_loaded: bool
    checkpoint_info: Optional[Dict[str, Any]]


class ExampleResult(BaseModel):
    """Schema for a single pre-computed example result."""
    image_id: int
    attack_type: str
    epsilon: float
    original_class: str
    adversarial_class: str
    attack_success: bool
    original_confidence: float
    adversarial_confidence: float


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Adversarial ML API",
    description="REST API for generating and analyzing adversarial examples using FGSM and PGD attacks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# CORS Configuration
# ============================================================================

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State
# ============================================================================

class ModelManager:
    """
    Singleton class for managing model and attack instances.

    This class ensures the model is loaded only once and provides
    access to attack implementations.
    """

    def __init__(self):
        """Initialize the model manager."""
        self.model = None
        self.checkpoint_info = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        """
        Load the trained model from checkpoint.

        Attempts to load the model from the configured path. If loading fails,
        the model remains uninitialized and endpoints will return appropriate errors.
        """
        try:
            print(f"Loading model from {MODEL_PATH}...")

            # Create model
            self.model = get_resnet18(
                num_classes=len(CIFAR10_CLASSES),
                device=DEVICE
            )

            # Load checkpoint if it exists
            if os.path.exists(MODEL_PATH):
                info = load_checkpoint(
                    MODEL_PATH,
                    self.model,
                    device=DEVICE
                )
                self.checkpoint_info = {
                    'epoch': info.get('epoch', 'unknown'),
                    'accuracy': info.get('accuracy', 'unknown'),
                    'loss': info.get('loss', 'unknown')
                }
                print(f"✓ Model loaded successfully from epoch {info.get('epoch', 'unknown')}")
                print(f"  Accuracy: {info.get('accuracy', 'unknown')}")
            else:
                print(f"⚠ Model checkpoint not found at {MODEL_PATH}")
                print("  Model initialized with random weights")
                self.checkpoint_info = None

            self.model.eval()  # Set to evaluation mode
            self.model_loaded = True

        except Exception as e:
            print(f"✗ Failed to load model: {str(e)}")
            print(traceback.format_exc())
            self.model_loaded = False

    def get_model(self) -> nn.Module:
        """
        Get the loaded model.

        Returns:
            nn.Module: The loaded PyTorch model

        Raises:
            HTTPException: If model is not loaded
        """
        if not self.model_loaded or self.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs."
            )
        return self.model

    def create_attack(self, attack_type: str, **kwargs):
        """
        Create an attack instance.

        Args:
            attack_type: Type of attack ('fgsm' or 'pgd')
            **kwargs: Attack-specific parameters

        Returns:
            Attack instance (FGSM or PGD)

        Raises:
            ValueError: If attack_type is invalid
        """
        model = self.get_model()

        if attack_type == 'fgsm':
            return FGSM(
                model=model,
                epsilon=kwargs.get('epsilon', 0.03),
                targeted=kwargs.get('targeted', False)
            )
        elif attack_type == 'pgd':
            return PGD(
                model=model,
                epsilon=kwargs.get('epsilon', 0.03),
                alpha=kwargs.get('alpha', 0.01),
                iterations=kwargs.get('iterations', 20),
                random_start=kwargs.get('random_start', True),
                targeted=kwargs.get('targeted', False)
            )
        else:
            raise ValueError(f"Invalid attack type: {attack_type}")


# Initialize model manager
model_manager = ModelManager()

# ============================================================================
# Utility Functions
# ============================================================================

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for model input.

    Args:
        image: PIL Image (RGB)

    Returns:
        torch.Tensor: Preprocessed image tensor (1, 3, 32, 32)
    """
    # Resize to 32x32 (CIFAR-10 size)
    image = image.resize((32, 32), Image.BILINEAR)

    # Convert to tensor and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0

    # Apply CIFAR-10 normalization
    for i in range(3):
        image_array[:, :, i] = (image_array[:, :, i] - MEAN[i]) / STD[i]

    # Convert to tensor (C, H, W)
    image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(DEVICE)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize an image tensor for visualization.

    Args:
        tensor: Normalized image tensor (C, H, W) or (1, C, H, W)

    Returns:
        np.ndarray: Denormalized image (H, W, C) in range [0, 255]
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Move to CPU and convert to numpy
    image = tensor.cpu().detach().numpy().transpose(1, 2, 0)

    # Denormalize
    for i in range(3):
        image[:, :, i] = image[:, :, i] * STD[i] + MEAN[i]

    # Clip to [0, 1] and scale to [0, 255]
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)

    return image


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """
    Convert image tensor to base64-encoded PNG.

    Args:
        tensor: Image tensor (C, H, W) or (1, C, H, W)

    Returns:
        str: Base64-encoded PNG image
    """
    # Denormalize and convert to numpy
    image = denormalize_image(tensor)

    # Convert to PIL Image
    pil_image = Image.fromarray(image, mode='RGB')

    # Encode as PNG
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")

    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


def create_perturbation_visualization(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    amplification: float = 10.0
) -> str:
    """
    Create a visualization of the perturbation.

    Args:
        original: Original image tensor
        adversarial: Adversarial image tensor
        amplification: Amplification factor for visualization

    Returns:
        str: Base64-encoded perturbation visualization
    """
    # Compute perturbation
    perturbation = (adversarial - original).squeeze(0)

    # Amplify for visibility
    perturbation = perturbation * amplification

    # Normalize to [0, 1]
    perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)

    # Convert to numpy
    pert_np = perturbation.cpu().detach().numpy().transpose(1, 2, 0)
    pert_np = (pert_np * 255).astype(np.uint8)

    # Convert to PIL and encode
    pil_image = Image.fromarray(pert_np, mode='RGB')
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


def predict(model: nn.Module, image: torch.Tensor) -> PredictionResponse:
    """
    Make a prediction on an image.

    Args:
        model: PyTorch model
        image: Preprocessed image tensor

    Returns:
        PredictionResponse: Prediction results
    """
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

        # Create probability dictionary
        all_probs = {
            CIFAR10_CLASSES[i]: float(probabilities[i].item())
            for i in range(len(CIFAR10_CLASSES))
        }

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=CIFAR10_CLASSES[predicted_class],
        confidence=confidence,
        all_probabilities=all_probs
    )


def compute_perturbation_stats(
    original: torch.Tensor,
    adversarial: torch.Tensor
) -> Dict[str, float]:
    """
    Compute statistics about the perturbation.

    Args:
        original: Original image tensor
        adversarial: Adversarial image tensor

    Returns:
        dict: Perturbation statistics (L0, L1, L2, Linf norms)
    """
    perturbation = (adversarial - original).squeeze(0)

    # Compute various norms
    l0_norm = torch.count_nonzero(perturbation).item()
    l1_norm = torch.sum(torch.abs(perturbation)).item()
    l2_norm = torch.norm(perturbation, p=2).item()
    linf_norm = torch.max(torch.abs(perturbation)).item()

    return {
        'l0_norm': float(l0_norm),
        'l1_norm': float(l1_norm),
        'l2_norm': float(l2_norm),
        'linf_norm': float(linf_norm),
        'mean_perturbation': float(torch.mean(torch.abs(perturbation)).item()),
        'max_perturbation': float(linf_norm)
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint providing API information.

    Returns:
        dict: API metadata and available endpoints
    """
    return {
        "name": "Adversarial ML API",
        "version": "1.0.0",
        "description": "REST API for adversarial example generation",
        "endpoints": {
            "GET /": "API information",
            "GET /model-info": "Get model information and statistics",
            "POST /upload": "Upload and classify an image",
            "POST /attack": "Generate adversarial example",
            "GET /example-results": "Get pre-computed example results",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "Alternative API documentation"
        },
        "documentation": {
            "interactive_docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the loaded model.

    Returns:
        ModelInfoResponse: Model architecture and performance information

    Example:
        ```bash
        curl http://localhost:8000/model-info
        ```
    """
    try:
        model = model_manager.get_model()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())

        return ModelInfoResponse(
            model_name="ResNet18",
            num_classes=len(CIFAR10_CLASSES),
            class_names=CIFAR10_CLASSES,
            parameters=total_params,
            device=str(DEVICE),
            model_loaded=model_manager.model_loaded,
            checkpoint_info=model_manager.checkpoint_info
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model info: {str(e)}"
        )


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image and get model predictions.

    Args:
        file: Image file (PNG, JPG, JPEG)

    Returns:
        dict: Prediction results and image information

    Raises:
        HTTPException: If image processing fails

    Example:
        ```bash
        curl -X POST -F "file=@image.png" http://localhost:8000/upload
        ```
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (PNG, JPG, JPEG)"
            )

        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Preprocess
        image_tensor = preprocess_image(image)

        # Get model and predict
        model = model_manager.get_model()
        prediction = predict(model, image_tensor)

        # Encode original image
        original_b64 = tensor_to_base64(image_tensor)

        return {
            "filename": file.filename,
            "size": image.size,
            "prediction": prediction.dict(),
            "image": original_b64,
            "message": "Image uploaded and classified successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/attack", response_model=AttackResponse)
async def generate_attack(
    file: UploadFile = File(...),
    attack_type: str = Form('fgsm'),
    epsilon: float = Form(0.03),
    targeted: bool = Form(False),
    target_class: Optional[int] = Form(None),
    pgd_alpha: float = Form(0.01),
    pgd_iterations: int = Form(20),
    random_start: bool = Form(True)
):
    """
    Generate an adversarial example for an uploaded image.

    Args:
        file: Image file to attack
        attack_type: Type of attack ('fgsm' or 'pgd')
        epsilon: Perturbation magnitude (0.0-1.0)
        targeted: Perform targeted attack
        target_class: Target class for targeted attack (0-9)
        pgd_alpha: Step size for PGD
        pgd_iterations: Number of PGD iterations
        random_start: Use random start for PGD

    Returns:
        AttackResponse: Complete attack results with images

    Example:
        ```bash
        curl -X POST \
          -F "file=@image.png" \
          -F "attack_type=pgd" \
          -F "epsilon=0.03" \
          http://localhost:8000/attack
        ```
    """
    try:
        # Validate parameters
        if epsilon < 0 or epsilon > 1:
            raise HTTPException(
                status_code=400,
                detail="epsilon must be between 0 and 1"
            )

        if attack_type not in ['fgsm', 'pgd']:
            raise HTTPException(
                status_code=400,
                detail="attack_type must be 'fgsm' or 'pgd'"
            )

        if targeted and (target_class is None or target_class < 0 or target_class > 9):
            raise HTTPException(
                status_code=400,
                detail="target_class must be specified (0-9) for targeted attacks"
            )

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = preprocess_image(image)

        # Get model and original prediction
        model = model_manager.get_model()
        original_pred = predict(model, image_tensor)

        # Determine true label (use predicted label for untargeted, target for targeted)
        if targeted:
            labels = torch.tensor([target_class]).to(DEVICE)
        else:
            labels = torch.tensor([original_pred.predicted_class]).to(DEVICE)

        # Create attack
        attack_params = {
            'epsilon': epsilon,
            'targeted': targeted
        }

        if attack_type == 'pgd':
            attack_params.update({
                'alpha': pgd_alpha,
                'iterations': pgd_iterations,
                'random_start': random_start
            })

        attack = model_manager.create_attack(attack_type, **attack_params)

        # Generate adversarial example
        adversarial_tensor = attack.generate(image_tensor, labels)

        # Get adversarial prediction
        adversarial_pred = predict(model, adversarial_tensor)

        # Determine attack success
        if targeted:
            success = (adversarial_pred.predicted_class == target_class)
        else:
            success = (adversarial_pred.predicted_class != original_pred.predicted_class)

        # Create visualizations
        original_b64 = tensor_to_base64(image_tensor)
        adversarial_b64 = tensor_to_base64(adversarial_tensor)
        perturbation_b64 = create_perturbation_visualization(
            image_tensor, adversarial_tensor
        )

        # Compute perturbation statistics
        pert_stats = compute_perturbation_stats(image_tensor, adversarial_tensor)

        # Prepare response
        return AttackResponse(
            success=success,
            original_prediction=original_pred,
            adversarial_prediction=adversarial_pred,
            original_image=original_b64,
            adversarial_image=adversarial_b64,
            perturbation=perturbation_b64,
            attack_parameters={
                'attack_type': attack_type,
                'epsilon': epsilon,
                'targeted': targeted,
                'target_class': target_class,
                **({
                    'pgd_alpha': pgd_alpha,
                    'pgd_iterations': pgd_iterations,
                    'random_start': random_start
                } if attack_type == 'pgd' else {})
            },
            perturbation_stats=pert_stats
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating attack: {str(e)}\n{traceback.format_exc()}"
        )


@app.post("/batch-attack")
async def generate_batch_attack(
    files: List[UploadFile] = File(...),
    attack_type: str = Form('fgsm'),
    epsilon: float = Form(0.03),
    pgd_alpha: float = Form(0.01),
    pgd_iterations: int = Form(20),
    random_start: bool = Form(True),
    export_format: str = Form('json')  # 'json' or 'csv'
):
    """
    Generate adversarial examples for multiple images in batch.

    Args:
        files: List of image files to attack
        attack_type: Type of attack ('fgsm' or 'pgd')
        epsilon: Perturbation magnitude (0.0-1.0)
        pgd_alpha: Step size for PGD
        pgd_iterations: Number of PGD iterations
        random_start: Use random start for PGD
        export_format: Export format ('json' or 'csv')

    Returns:
        Batch attack results with aggregate statistics

    Example:
        ```bash
        curl -X POST \
          -F "files=@image1.png" \
          -F "files=@image2.png" \
          -F "files=@image3.png" \
          -F "attack_type=pgd" \
          -F "epsilon=0.03" \
          http://localhost:8000/batch-attack
        ```
    """
    try:
        # Validate parameters
        if epsilon < 0 or epsilon > 1:
            raise HTTPException(
                status_code=400,
                detail="epsilon must be between 0 and 1"
            )

        if attack_type not in ['fgsm', 'pgd']:
            raise HTTPException(
                status_code=400,
                detail="attack_type must be 'fgsm' or 'pgd'"
            )

        if export_format not in ['json', 'csv']:
            raise HTTPException(
                status_code=400,
                detail="export_format must be 'json' or 'csv'"
            )

        if len(files) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one file must be provided"
            )

        if len(files) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 files allowed per batch"
            )

        # Get model
        model = model_manager.get_model()

        # Create attack
        attack_params = {
            'epsilon': epsilon,
            'targeted': False
        }

        if attack_type == 'pgd':
            attack_params.update({
                'alpha': pgd_alpha,
                'iterations': pgd_iterations,
                'random_start': random_start
            })

        attack = model_manager.create_attack(attack_type, **attack_params)

        # Process all images
        results = []
        total_success = 0
        total_linf = 0.0
        total_l2 = 0.0
        total_l1 = 0.0
        class_accuracy = {cls: {'total': 0, 'correct': 0, 'attacked': 0}
                          for cls in CIFAR10_CLASSES}

        import time
        start_time = time.time()

        for idx, file in enumerate(files):
            try:
                # Read and process image
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert('RGB')
                image_tensor = preprocess_image(image)

                # Get original prediction
                original_pred = predict(model, image_tensor)
                labels = torch.tensor([original_pred.predicted_class]).to(DEVICE)

                # Generate adversarial example
                adversarial_tensor = attack.generate(image_tensor, labels)

                # Get adversarial prediction
                adversarial_pred = predict(model, adversarial_tensor)

                # Compute statistics
                success = (adversarial_pred.predicted_class != original_pred.predicted_class)
                pert_stats = compute_perturbation_stats(image_tensor, adversarial_tensor)

                # Update aggregates
                if success:
                    total_success += 1
                total_linf += pert_stats['linf_norm']
                total_l2 += pert_stats['l2_norm']
                total_l1 += pert_stats['l1_norm']

                # Update class-wise stats
                orig_class_name = CIFAR10_CLASSES[original_pred.predicted_class]
                class_accuracy[orig_class_name]['total'] += 1
                if success:
                    class_accuracy[orig_class_name]['attacked'] += 1
                else:
                    class_accuracy[orig_class_name]['correct'] += 1

                # Store result
                result = {
                    'image_id': idx,
                    'filename': file.filename,
                    'original_class': CIFAR10_CLASSES[original_pred.predicted_class],
                    'original_confidence': float(original_pred.confidence),
                    'adversarial_class': CIFAR10_CLASSES[adversarial_pred.predicted_class],
                    'adversarial_confidence': float(adversarial_pred.confidence),
                    'attack_success': success,
                    'linf_norm': float(pert_stats['linf_norm']),
                    'l2_norm': float(pert_stats['l2_norm']),
                    'l1_norm': float(pert_stats['l1_norm']),
                    'l0_norm': int(pert_stats['l0_norm'])
                }
                results.append(result)

            except Exception as e:
                # Add error result for this image
                results.append({
                    'image_id': idx,
                    'filename': file.filename,
                    'error': str(e),
                    'attack_success': False
                })

        end_time = time.time()
        processing_time = end_time - start_time

        # Compute aggregate statistics
        num_images = len(results)
        valid_results = [r for r in results if 'error' not in r]
        num_valid = len(valid_results)

        if num_valid > 0:
            attack_success_rate = (total_success / num_valid) * 100
            avg_linf = total_linf / num_valid
            avg_l2 = total_l2 / num_valid
            avg_l1 = total_l1 / num_valid
        else:
            attack_success_rate = 0.0
            avg_linf = avg_l2 = avg_l1 = 0.0

        # Prepare response
        response = {
            'summary': {
                'total_images': num_images,
                'successful_attacks': total_success,
                'failed_attacks': num_valid - total_success,
                'errors': num_images - num_valid,
                'attack_success_rate': round(attack_success_rate, 2),
                'average_linf_norm': round(avg_linf, 6),
                'average_l2_norm': round(avg_l2, 6),
                'average_l1_norm': round(avg_l1, 6),
                'processing_time_seconds': round(processing_time, 2),
                'images_per_second': round(num_images / processing_time, 2)
            },
            'attack_parameters': {
                'attack_type': attack_type,
                'epsilon': epsilon,
                'pgd_alpha': pgd_alpha if attack_type == 'pgd' else None,
                'pgd_iterations': pgd_iterations if attack_type == 'pgd' else None,
                'random_start': random_start if attack_type == 'pgd' else None
            },
            'class_wise_statistics': {
                cls: {
                    'total': stats['total'],
                    'robust': stats['correct'],
                    'vulnerable': stats['attacked'],
                    'robustness_rate': round((stats['correct'] / stats['total'] * 100)
                                             if stats['total'] > 0 else 0, 2)
                }
                for cls, stats in class_accuracy.items()
                if stats['total'] > 0
            },
            'results': results
        }

        # Export as CSV if requested
        if export_format == 'csv':
            import csv
            from io import StringIO

            output = StringIO()
            if valid_results:
                writer = csv.DictWriter(output, fieldnames=valid_results[0].keys())
                writer.writeheader()
                writer.writerows(valid_results)

            return JSONResponse(content={
                **response,
                'csv_export': output.getvalue()
            })

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in batch attack: {str(e)}\n{traceback.format_exc()}"
        )


@app.post("/attack-iterations")
async def generate_attack_with_iterations(
    file: UploadFile = File(...),
    epsilon: float = Form(0.03),
    pgd_alpha: float = Form(0.0075),
    pgd_iterations: int = Form(20),
    random_start: bool = Form(True)
):
    """
    Generate PGD attack and return all intermediate iterations for visualization.

    This endpoint is specifically designed for real-time attack visualization,
    returning the adversarial image at each iteration step.

    Args:
        file: Image file to attack
        epsilon: Perturbation magnitude (0.0-1.0)
        pgd_alpha: Step size for PGD
        pgd_iterations: Number of PGD iterations
        random_start: Use random start for PGD

    Returns:
        List of iterations with images and predictions at each step
    """
    try:
        # Validate parameters
        if epsilon < 0 or epsilon > 1:
            raise HTTPException(
                status_code=400,
                detail="epsilon must be between 0 and 1"
            )

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = preprocess_image(image)

        # Get model and original prediction
        model = model_manager.get_model()
        original_pred = predict(model, image_tensor)
        labels = torch.tensor([original_pred.predicted_class]).to(DEVICE)

        # Initialize iteration storage
        iterations = []

        # Store original image
        iterations.append({
            'iteration': 0,
            'image': tensor_to_base64(image_tensor),
            'predicted_class': CIFAR10_CLASSES[original_pred.predicted_class],
            'confidence': float(original_pred.confidence),
            'perturbation_linf': 0.0,
            'is_adversarial': False
        })

        # Generate PGD attack with iteration tracking
        # Initialize adversarial image
        if random_start:
            perturbation = torch.empty_like(image_tensor).uniform_(-epsilon, epsilon)
            adversarial_tensor = torch.clamp(image_tensor + perturbation, 0, 1)
        else:
            adversarial_tensor = image_tensor.clone()

        # Iterative attack
        for i in range(pgd_iterations):
            adversarial_tensor.requires_grad = True

            # Forward pass
            output = model(adversarial_tensor)

            # Calculate loss
            loss = torch.nn.functional.cross_entropy(output, labels)

            # Backward pass
            model.zero_grad()
            loss.backward()

            # Update with gradient sign
            gradient_sign = adversarial_tensor.grad.sign()
            adversarial_tensor = adversarial_tensor + pgd_alpha * gradient_sign

            # Project back to epsilon-ball
            perturbation = torch.clamp(
                adversarial_tensor - image_tensor,
                -epsilon,
                epsilon
            )
            adversarial_tensor = torch.clamp(image_tensor + perturbation, 0, 1)
            adversarial_tensor = adversarial_tensor.detach()

            # Get prediction at this iteration
            with torch.no_grad():
                iter_pred = predict(model, adversarial_tensor)

            # Compute current perturbation
            current_pert = (adversarial_tensor - image_tensor).abs()
            current_linf = float(current_pert.max())

            # Check if adversarial
            is_adversarial = (iter_pred.predicted_class != original_pred.predicted_class)

            # Store iteration
            iterations.append({
                'iteration': i + 1,
                'image': tensor_to_base64(adversarial_tensor),
                'predicted_class': CIFAR10_CLASSES[iter_pred.predicted_class],
                'confidence': float(iter_pred.confidence),
                'perturbation_linf': current_linf,
                'is_adversarial': is_adversarial
            })

        # Return all iterations
        return JSONResponse(content={
            'original_class': CIFAR10_CLASSES[original_pred.predicted_class],
            'original_confidence': float(original_pred.confidence),
            'total_iterations': pgd_iterations,
            'epsilon': epsilon,
            'alpha': pgd_alpha,
            'iterations': iterations,
            'final_success': iterations[-1]['is_adversarial']
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating attack iterations: {str(e)}\n{traceback.format_exc()}"
        )


@app.get("/example-results", response_model=List[ExampleResult])
async def get_example_results():
    """
    Get pre-computed example attack results.

    This endpoint returns a list of pre-computed adversarial examples
    from the evaluation experiments, demonstrating attack effectiveness
    across different configurations.

    Returns:
        List[ExampleResult]: List of example attack results

    Example:
        ```bash
        curl http://localhost:8000/example-results
        ```
    """
    try:
        # Check if evaluation results exist
        results_path = os.path.join('results', 'attack_evaluation', 'attack_results.json')

        if not os.path.exists(results_path):
            # Return dummy examples if no results file exists
            return [
                ExampleResult(
                    image_id=1,
                    attack_type="FGSM",
                    epsilon=0.03,
                    original_class="cat",
                    adversarial_class="dog",
                    attack_success=True,
                    original_confidence=0.92,
                    adversarial_confidence=0.67
                ),
                ExampleResult(
                    image_id=2,
                    attack_type="PGD-20",
                    epsilon=0.03,
                    original_class="airplane",
                    adversarial_class="bird",
                    attack_success=True,
                    original_confidence=0.95,
                    adversarial_confidence=0.78
                ),
                ExampleResult(
                    image_id=3,
                    attack_type="FGSM",
                    epsilon=0.01,
                    original_class="ship",
                    adversarial_class="ship",
                    attack_success=False,
                    original_confidence=0.89,
                    adversarial_confidence=0.85
                )
            ]

        # Load actual results if available
        with open(results_path, 'r') as f:
            results_data = json.load(f)

        # Extract summary statistics
        examples = []
        example_id = 1

        # FGSM examples
        if 'fgsm' in results_data:
            for eps, data in results_data['fgsm'].items():
                examples.append(ExampleResult(
                    image_id=example_id,
                    attack_type="FGSM",
                    epsilon=float(eps),
                    original_class="various",
                    adversarial_class="various",
                    attack_success=data['attack_success_rate'] > 0.5,
                    original_confidence=data['clean_accuracy'],
                    adversarial_confidence=data['adversarial_accuracy']
                ))
                example_id += 1

        # PGD examples
        if 'pgd' in results_data:
            for config_name, data in list(results_data['pgd'].items())[:3]:
                examples.append(ExampleResult(
                    image_id=example_id,
                    attack_type=config_name,
                    epsilon=data.get('epsilon', 0.03),
                    original_class="various",
                    adversarial_class="various",
                    attack_success=data['attack_success_rate'] > 0.5,
                    original_confidence=data['clean_accuracy'],
                    adversarial_confidence=data['adversarial_accuracy']
                ))
                example_id += 1

        return examples

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading example results: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns:
        dict: Service health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_manager.model_loaded,
        "device": str(DEVICE)
    }


# ============================================================================
# Application Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.

    Logs startup information and model status.
    """
    print("=" * 80)
    print("ADVERSARIAL ML API STARTING")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Model loaded: {model_manager.model_loaded}")
    if model_manager.checkpoint_info:
        print(f"Checkpoint info: {model_manager.checkpoint_info}")
    print("=" * 80)
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler."""
    print("Shutting down Adversarial ML API...")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
