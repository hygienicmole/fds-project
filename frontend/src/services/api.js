/**
 * API Service for Adversarial ML Backend
 *
 * Provides a centralized interface for all API calls to the FastAPI backend.
 * Handles request/response formatting, error handling, and base64 image decoding.
 */

import axios from 'axios';

// API base URL - can be configured via environment variable
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds timeout for attack generation
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API] Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log(`[API] Response from ${response.config.url}:`, response.status);
    return response;
  },
  (error) => {
    console.error('[API] Response error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * API Service Object
 */
const api = {
  /**
   * Get API root information
   * @returns {Promise<Object>} API metadata
   */
  getRoot: async () => {
    const response = await apiClient.get('/');
    return response.data;
  },

  /**
   * Get model information and statistics
   * @returns {Promise<Object>} Model info including architecture, parameters, accuracy
   */
  getModelInfo: async () => {
    const response = await apiClient.get('/model-info');
    return response.data;
  },

  /**
   * Upload and classify an image
   * @param {File} file - Image file to upload
   * @returns {Promise<Object>} Classification results with prediction and confidence
   */
  uploadImage: async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },

  /**
   * Generate adversarial attack on an image
   * @param {File} file - Image file to attack
   * @param {Object} params - Attack parameters
   * @param {string} params.attack_type - 'fgsm' or 'pgd'
   * @param {number} params.epsilon - Perturbation magnitude (0-1)
   * @param {boolean} params.targeted - Whether to perform targeted attack
   * @param {number} params.target_class - Target class for targeted attack (0-9)
   * @param {number} params.pgd_alpha - PGD step size
   * @param {number} params.pgd_iterations - PGD number of iterations
   * @param {boolean} params.random_start - Use random start for PGD
   * @returns {Promise<Object>} Attack results with original and adversarial images
   */
  generateAttack: async (file, params = {}) => {
    const formData = new FormData();
    formData.append('file', file);

    // Add attack parameters
    const defaultParams = {
      attack_type: 'fgsm',
      epsilon: 0.03,
      targeted: false,
      pgd_alpha: 0.01,
      pgd_iterations: 20,
      random_start: true,
    };

    const attackParams = { ...defaultParams, ...params };

    Object.entries(attackParams).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        formData.append(key, value);
      }
    });

    const response = await apiClient.post('/attack', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },

  /**
   * Generate batch adversarial attacks on multiple images
   * @param {Array<File>} files - Array of image files to attack
   * @param {Object} params - Attack parameters
   * @param {string} params.attack_type - 'fgsm' or 'pgd'
   * @param {number} params.epsilon - Perturbation magnitude (0-1)
   * @param {number} params.pgd_alpha - PGD step size
   * @param {number} params.pgd_iterations - PGD number of iterations
   * @param {boolean} params.random_start - Use random start for PGD
   * @param {string} params.export_format - 'json' or 'csv'
   * @returns {Promise<Object>} Batch attack results with summary and individual results
   */
  generateBatchAttack: async (files, params = {}) => {
    const formData = new FormData();

    // Add all files
    files.forEach((file) => {
      formData.append('files', file);
    });

    // Add attack parameters
    const defaultParams = {
      attack_type: 'fgsm',
      epsilon: 0.03,
      pgd_alpha: 0.01,
      pgd_iterations: 20,
      random_start: true,
      export_format: 'json',
    };

    const attackParams = { ...defaultParams, ...params };

    Object.entries(attackParams).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        formData.append(key, value);
      }
    });

    const response = await apiClient.post('/batch-attack', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 180000, // 3 minutes for batch processing
    });

    return response.data;
  },

  /**
   * Generate PGD attack with iteration-by-iteration results for visualization
   * @param {File} file - Image file to attack
   * @param {Object} params - Attack parameters
   * @param {number} params.epsilon - Perturbation magnitude (0-1)
   * @param {number} params.pgd_alpha - PGD step size
   * @param {number} params.pgd_iterations - PGD number of iterations
   * @param {boolean} params.random_start - Use random start for PGD
   * @returns {Promise<Object>} Attack iterations with images and predictions at each step
   */
  generateAttackIterations: async (file, params = {}) => {
    const formData = new FormData();
    formData.append('file', file);

    const defaultParams = {
      epsilon: 0.03,
      pgd_alpha: 0.0075,
      pgd_iterations: 20,
      random_start: true,
    };

    const attackParams = { ...defaultParams, ...params };

    Object.entries(attackParams).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        formData.append(key, value);
      }
    });

    const response = await apiClient.post('/attack-iterations', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 120000, // 2 minutes for iteration tracking
    });

    return response.data;
  },

  /**
   * Get pre-computed example attack results
   * @returns {Promise<Array>} List of example attack results
   */
  getExampleResults: async () => {
    const response = await apiClient.get('/example-results');
    return response.data;
  },

  /**
   * Check API health status
   * @returns {Promise<Object>} Health status information
   */
  healthCheck: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },

  /**
   * Get random CIFAR-10 test set samples
   * @param {number} numSamples - Number of samples to fetch (default: 20, max: 50)
   * @returns {Promise<Object>} CIFAR-10 samples with images and labels
   */
  getCifarSamples: async (numSamples = 20) => {
    const response = await apiClient.get('/cifar-samples', {
      params: { num_samples: numSamples },
    });
    return response.data;
  },
};

/**
 * Utility functions for image handling
 */
export const imageUtils = {
  /**
   * Convert base64 string to blob
   * @param {string} base64 - Base64 encoded image
   * @returns {Blob} Image blob
   */
  base64ToBlob: (base64) => {
    // Remove data URL prefix if present
    const base64Data = base64.replace(/^data:image\/\w+;base64,/, '');
    const byteCharacters = atob(base64Data);
    const byteNumbers = new Array(byteCharacters.length);

    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }

    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: 'image/png' });
  },

  /**
   * Download image from base64 string
   * @param {string} base64 - Base64 encoded image
   * @param {string} filename - Download filename
   */
  downloadImage: (base64, filename = 'image.png') => {
    const blob = imageUtils.base64ToBlob(base64);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  },

  /**
   * Validate image file
   * @param {File} file - File to validate
   * @returns {Object} Validation result with isValid and error message
   */
  validateImageFile: (file) => {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!file) {
      return { isValid: false, error: 'No file provided' };
    }

    if (!validTypes.includes(file.type)) {
      return { isValid: false, error: 'Invalid file type. Please upload a JPEG or PNG image.' };
    }

    if (file.size > maxSize) {
      return { isValid: false, error: 'File too large. Maximum size is 10MB.' };
    }

    return { isValid: true, error: null };
  },
};

/**
 * CIFAR-10 class names
 */
export const CIFAR10_CLASSES = [
  'airplane',
  'automobile',
  'bird',
  'cat',
  'deer',
  'dog',
  'frog',
  'horse',
  'ship',
  'truck',
];

/**
 * Attack type configurations
 */
export const ATTACK_CONFIGS = {
  fgsm: {
    name: 'FGSM',
    fullName: 'Fast Gradient Sign Method',
    description: 'One-step attack using gradient sign',
    defaultParams: {
      epsilon: 0.03,
    },
  },
  pgd: {
    name: 'PGD',
    fullName: 'Projected Gradient Descent',
    description: 'Iterative attack with projection',
    defaultParams: {
      epsilon: 0.03,
      alpha: 0.01,
      iterations: 20,
      random_start: true,
    },
  },
};

export default api;
