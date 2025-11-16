import { useState, useEffect } from 'react';
import {
  Upload,
  Zap,
  Download,
  Info,
  AlertCircle,
  CheckCircle,
  ImageIcon,
  Settings,
  Eye,
  TrendingUp,
} from 'lucide-react';
import api, { imageUtils, CIFAR10_CLASSES, ATTACK_CONFIGS } from '../services/api';
import { formatConfidence, getConfidenceColor, formatEpsilon } from '../utils/helpers';
import Loading from './Loading';
import ErrorMessage from './ErrorMessage';

/**
 * AttackDemo Component
 *
 * Interactive component for generating and visualizing adversarial attacks.
 * Provides comprehensive controls for attack configuration and detailed results display.
 *
 * Features:
 * - Image upload or sample selection
 * - FGSM and PGD attack configuration
 * - Real-time parameter adjustment
 * - Side-by-side result comparison
 * - Perturbation visualization
 * - Educational tooltips
 * - Download functionality
 */
const AttackDemo = () => {
  // State for image handling
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [useSample, setUseSample] = useState(false);

  // State for attack configuration
  const [attackType, setAttackType] = useState('fgsm');
  const [epsilon, setEpsilon] = useState(0.03);
  const [pgdIterations, setPgdIterations] = useState(20);
  const [pgdAlpha, setPgdAlpha] = useState(0.01);
  const [randomStart, setRandomStart] = useState(true);

  // State for targeted attacks
  const [targeted, setTargeted] = useState(false);
  const [targetClass, setTargetClass] = useState(0);

  // State for UI
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [showInfo, setShowInfo] = useState(false);

  // Auto-calculate alpha as epsilon/4 when epsilon changes (optional)
  useEffect(() => {
    setPgdAlpha(epsilon * 0.25);
  }, [epsilon]);

  /**
   * Handle file selection from user upload
   */
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    const validation = imageUtils.validateImageFile(selectedFile);
    if (!validation.isValid) {
      setError(validation.error);
      return;
    }

    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setUseSample(false);
    setError(null);
    setResult(null);
  };

  /**
   * Handle sample image selection (simulated)
   */
  const handleSampleSelect = () => {
    // In a real implementation, you would fetch actual CIFAR-10 test images
    setUseSample(true);
    setFile(null);
    setPreviewUrl(null);
    setError('Sample image selection not implemented. Please upload an image instead.');
    setResult(null);
  };

  /**
   * Generate adversarial attack
   */
  const handleGenerateAttack = async () => {
    if (!file && !useSample) {
      setError('Please select or upload an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const params = {
        attack_type: attackType,
        epsilon: parseFloat(epsilon),
        targeted,
        random_start: randomStart,
      };

      // Add PGD-specific parameters
      if (attackType === 'pgd') {
        params.pgd_iterations = parseInt(pgdIterations);
        params.pgd_alpha = parseFloat(pgdAlpha);
      }

      // Add target class for targeted attacks
      if (targeted) {
        params.target_class = parseInt(targetClass);
      }

      const response = await api.generateAttack(file, params);
      setResult(response);
    } catch (err) {
      const errorMsg = err.response?.data?.detail ||
                      'Failed to generate attack. Please check your backend server is running.';
      setError(errorMsg);
      console.error('Attack generation error:', err);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Download image
   */
  const handleDownload = (imageData, filename) => {
    imageUtils.downloadImage(imageData, filename);
  };

  /**
   * Reset all state
   */
  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setUseSample(false);
    setResult(null);
    setError(null);
    setAttackType('fgsm');
    setEpsilon(0.03);
    setPgdIterations(20);
    setPgdAlpha(0.01);
    setTargeted(false);
    setTargetClass(0);
  };

  /**
   * Get attack description
   */
  const getAttackDescription = () => {
    const config = ATTACK_CONFIGS[attackType];
    return config ? config.description : '';
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header with Info Toggle */}
      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-3xl font-bold text-gray-900 mb-2">
            Interactive Attack Demo
          </h2>
          <p className="text-gray-600">
            Generate adversarial examples and observe model behavior
          </p>
        </div>
        <button
          onClick={() => setShowInfo(!showInfo)}
          className="btn-outline flex items-center"
        >
          <Info className="h-4 w-4 mr-2" />
          {showInfo ? 'Hide' : 'Show'} Info
        </button>
      </div>

      {/* Information Panel */}
      {showInfo && (
        <div className="card mb-6 bg-blue-50 border border-blue-200 animate-fade-in">
          <div className="flex items-start">
            <Info className="h-6 w-6 text-blue-600 mt-1 mr-3 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-blue-900 mb-2">How to Use</h3>
              <ol className="list-decimal list-inside space-y-1 text-sm text-blue-800">
                <li>Upload an image (JPEG or PNG, 32×32 recommended)</li>
                <li>Select attack type (FGSM or PGD)</li>
                <li>Adjust epsilon to control perturbation magnitude</li>
                <li>For PGD, configure iterations and step size</li>
                <li>Click "Generate Attack" and observe the results</li>
                <li>Download the adversarial image if desired</li>
              </ol>
            </div>
          </div>
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column - Controls */}
        <div className="space-y-6">
          {/* Image Upload Section */}
          <div className="card">
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <ImageIcon className="h-5 w-5 mr-2 text-primary-600" />
              1. Select Image
            </h3>

            <div className="space-y-4">
              {/* Upload Area */}
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-500 transition-colors cursor-pointer">
                <input
                  type="file"
                  accept="image/jpeg,image/jpg,image/png"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload-demo"
                />
                <label htmlFor="file-upload-demo" className="cursor-pointer block">
                  <Upload className="h-10 w-10 mx-auto text-gray-400 mb-3" />
                  <p className="text-gray-600 mb-1 font-medium">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-sm text-gray-500">
                    PNG or JPEG (recommended: 32×32, max 10MB)
                  </p>
                </label>
              </div>

              {/* Sample Selection (Placeholder) */}
              <div className="text-center">
                <p className="text-sm text-gray-500 mb-2">or</p>
                <button
                  onClick={handleSampleSelect}
                  className="btn-outline text-sm"
                  disabled
                >
                  Select from CIFAR-10 Test Set (Coming Soon)
                </button>
              </div>

              {/* Preview */}
              {previewUrl && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-sm font-medium text-gray-700 mb-2">Preview:</p>
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="w-32 h-32 object-contain mx-auto bg-white border border-gray-200 rounded"
                  />
                  <p className="text-xs text-gray-600 mt-2 text-center truncate">
                    {file?.name}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Attack Configuration */}
          <div className="card">
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <Settings className="h-5 w-5 mr-2 text-primary-600" />
              2. Configure Attack
            </h3>

            <div className="space-y-5">
              {/* Attack Type */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Attack Method
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setAttackType('fgsm')}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      attackType === 'fgsm'
                        ? 'border-primary-600 bg-primary-50 text-primary-700'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="font-semibold mb-1">FGSM</div>
                    <div className="text-xs text-gray-600">One-step attack</div>
                  </button>
                  <button
                    onClick={() => setAttackType('pgd')}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      attackType === 'pgd'
                        ? 'border-primary-600 bg-primary-50 text-primary-700'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="font-semibold mb-1">PGD</div>
                    <div className="text-xs text-gray-600">Iterative attack</div>
                  </button>
                </div>
                <p className="text-xs text-gray-600 mt-2">
                  {getAttackDescription()}
                </p>
              </div>

              {/* Epsilon Slider */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium text-gray-700">
                    Perturbation Magnitude (ε)
                  </label>
                  <span className="text-sm font-semibold text-primary-600 bg-primary-50 px-3 py-1 rounded">
                    {formatEpsilon(epsilon)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.001"
                  max="0.3"
                  step="0.001"
                  value={epsilon}
                  onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0.001 (subtle)</span>
                  <span>0.03 (standard)</span>
                  <span>0.3 (strong)</span>
                </div>
                <div className="mt-2 p-3 bg-gray-50 rounded text-xs text-gray-600">
                  <strong>Note:</strong> Larger epsilon = more visible perturbations but higher attack success rate
                </div>
              </div>

              {/* PGD-Specific Controls */}
              {attackType === 'pgd' && (
                <div className="space-y-4 pt-4 border-t border-gray-200 animate-fade-in">
                  {/* Iterations */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-gray-700">
                        Iterations
                      </label>
                      <span className="text-sm font-semibold text-primary-600 bg-primary-50 px-3 py-1 rounded">
                        {pgdIterations}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="100"
                      step="5"
                      value={pgdIterations}
                      onChange={(e) => setPgdIterations(parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>5 (fast)</span>
                      <span>20 (balanced)</span>
                      <span>100 (thorough)</span>
                    </div>
                  </div>

                  {/* Step Size (Alpha) */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-gray-700">
                        Step Size (α)
                      </label>
                      <span className="text-sm font-semibold text-primary-600 bg-primary-50 px-3 py-1 rounded">
                        {formatEpsilon(pgdAlpha)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.001"
                      max="0.05"
                      step="0.001"
                      value={pgdAlpha}
                      onChange={(e) => setPgdAlpha(parseFloat(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>0.001</span>
                      <span>ε/4 (recommended)</span>
                      <span>0.05</span>
                    </div>
                  </div>

                  {/* Random Start */}
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="random-start"
                      checked={randomStart}
                      onChange={(e) => setRandomStart(e.target.checked)}
                      className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                    />
                    <label htmlFor="random-start" className="ml-2 text-sm text-gray-700">
                      Random initialization (recommended for stronger attacks)
                    </label>
                  </div>
                </div>
              )}

              {/* Targeted Attack Option */}
              <div className="pt-4 border-t border-gray-200">
                <div className="flex items-center mb-3">
                  <input
                    type="checkbox"
                    id="targeted-attack"
                    checked={targeted}
                    onChange={(e) => setTargeted(e.target.checked)}
                    className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                  />
                  <label htmlFor="targeted-attack" className="ml-2 text-sm font-medium text-gray-700">
                    Targeted Attack
                  </label>
                </div>

                {targeted && (
                  <div className="animate-fade-in">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Target Class
                    </label>
                    <select
                      value={targetClass}
                      onChange={(e) => setTargetClass(parseInt(e.target.value))}
                      className="input-field text-sm"
                    >
                      {CIFAR10_CLASSES.map((className, index) => (
                        <option key={index} value={index}>
                          {index}: {className}
                        </option>
                      ))}
                    </select>
                    <p className="text-xs text-gray-600 mt-2">
                      The attack will try to misclassify the image as this class
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleGenerateAttack}
              disabled={(!file && !useSample) || loading}
              className="btn-primary flex-1 flex items-center justify-center"
            >
              {loading ? (
                <>
                  <div className="loader mr-2" style={{ width: '20px', height: '20px', borderWidth: '2px' }}></div>
                  Generating...
                </>
              ) : (
                <>
                  <Zap className="h-5 w-5 mr-2" />
                  Generate Attack
                </>
              )}
            </button>
            <button
              onClick={handleReset}
              className="btn-secondary"
              disabled={loading}
            >
              Reset
            </button>
          </div>
        </div>

        {/* Right Column - Results */}
        <div>
          {/* Error Display */}
          {error && <ErrorMessage message={error} onRetry={() => setError(null)} />}

          {/* Loading State */}
          {loading && (
            <div className="card">
              <Loading message={`Generating ${attackType.toUpperCase()} attack...`} />
              <p className="text-center text-sm text-gray-600 mt-4">
                This may take a few seconds for PGD attacks with many iterations
              </p>
            </div>
          )}

          {/* Results Display */}
          {result && !loading && (
            <div className="space-y-6 animate-fade-in">
              {/* Attack Summary Card */}
              <div className="card">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-semibold flex items-center">
                    <Eye className="h-5 w-5 mr-2 text-primary-600" />
                    3. Results
                  </h3>
                  <span className={`px-4 py-2 rounded-full text-sm font-semibold ${
                    result.success
                      ? 'bg-red-100 text-red-800'
                      : 'bg-green-100 text-green-800'
                  }`}>
                    {result.success ? (
                      <span className="flex items-center">
                        <AlertCircle className="h-4 w-4 mr-1" />
                        Attack Successful
                      </span>
                    ) : (
                      <span className="flex items-center">
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Attack Failed
                      </span>
                    )}
                  </span>
                </div>

                {/* Key Metrics Grid */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600 mb-1">Attack Type</div>
                    <div className="font-semibold text-gray-900">
                      {result.attack_parameters.attack_type.toUpperCase()}
                    </div>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600 mb-1">Epsilon (ε)</div>
                    <div className="font-semibold text-gray-900">
                      {formatEpsilon(result.attack_parameters.epsilon)}
                    </div>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600 mb-1">L∞ Norm</div>
                    <div className="font-semibold text-gray-900">
                      {result.perturbation_stats.linf_norm.toFixed(6)}
                    </div>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-600 mb-1">L2 Norm</div>
                    <div className="font-semibold text-gray-900">
                      {result.perturbation_stats.l2_norm.toFixed(4)}
                    </div>
                  </div>
                </div>

                {/* Perturbation Statistics */}
                <div className="border-t pt-4">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                    <TrendingUp className="h-4 w-4 mr-1" />
                    Perturbation Analysis
                  </h4>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Mean:</span>
                      <span className="font-mono">
                        {result.perturbation_stats.mean_perturbation.toFixed(6)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Max:</span>
                      <span className="font-mono">
                        {result.perturbation_stats.max_perturbation.toFixed(6)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">L0 Norm:</span>
                      <span className="font-mono">
                        {result.perturbation_stats.l0_norm.toFixed(0)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">L1 Norm:</span>
                      <span className="font-mono">
                        {result.perturbation_stats.l1_norm.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Side-by-Side Comparison */}
              <div className="card">
                <h4 className="font-semibold mb-4">Prediction Comparison</h4>
                <div className="grid grid-cols-2 gap-4">
                  {/* Original */}
                  <div className="space-y-2">
                    <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                      <div className="text-xs text-green-700 font-medium mb-1">
                        Original Prediction
                      </div>
                      <div className={`text-lg font-bold ${getConfidenceColor(result.original_prediction.confidence)}`}>
                        {result.original_prediction.predicted_label}
                      </div>
                      <div className="text-sm text-gray-600">
                        Confidence: {formatConfidence(result.original_prediction.confidence)}
                      </div>
                    </div>
                  </div>

                  {/* Adversarial */}
                  <div className="space-y-2">
                    <div className={`border rounded-lg p-3 ${
                      result.success
                        ? 'bg-red-50 border-red-200'
                        : 'bg-gray-50 border-gray-200'
                    }`}>
                      <div className={`text-xs font-medium mb-1 ${
                        result.success ? 'text-red-700' : 'text-gray-700'
                      }`}>
                        Adversarial Prediction
                      </div>
                      <div className={`text-lg font-bold ${getConfidenceColor(result.adversarial_prediction.confidence)}`}>
                        {result.adversarial_prediction.predicted_label}
                      </div>
                      <div className="text-sm text-gray-600">
                        Confidence: {formatConfidence(result.adversarial_prediction.confidence)}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Change Indicator */}
                {result.success && (
                  <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800">
                    <strong>Misclassification:</strong> The model changed its prediction from{' '}
                    <strong>{result.original_prediction.predicted_label}</strong> to{' '}
                    <strong>{result.adversarial_prediction.predicted_label}</strong>
                  </div>
                )}
              </div>

              {/* Images Display */}
              <div className="card">
                <h4 className="font-semibold mb-4">Visual Comparison</h4>
                <div className="space-y-6">
                  {/* Original Image */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-700">
                        Original Image
                      </span>
                      <button
                        onClick={() => handleDownload(result.original_image, 'original.png')}
                        className="text-primary-600 hover:text-primary-700 text-sm flex items-center transition-colors"
                      >
                        <Download className="h-4 w-4 mr-1" />
                        Download
                      </button>
                    </div>
                    <div className="bg-gray-100 p-4 rounded-lg">
                      <img
                        src={result.original_image}
                        alt="Original"
                        className="w-full max-w-xs mx-auto rounded border-2 border-gray-300 shadow-sm"
                      />
                    </div>
                  </div>

                  {/* Adversarial Image */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-700">
                        Adversarial Image
                      </span>
                      <button
                        onClick={() => handleDownload(result.adversarial_image, 'adversarial.png')}
                        className="text-primary-600 hover:text-primary-700 text-sm flex items-center transition-colors"
                      >
                        <Download className="h-4 w-4 mr-1" />
                        Download
                      </button>
                    </div>
                    <div className="bg-gray-100 p-4 rounded-lg">
                      <img
                        src={result.adversarial_image}
                        alt="Adversarial"
                        className="w-full max-w-xs mx-auto rounded border-2 border-red-300 shadow-sm"
                      />
                    </div>
                  </div>

                  {/* Perturbation */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-700">
                        Perturbation (10× amplified)
                      </span>
                      <button
                        onClick={() => handleDownload(result.perturbation, 'perturbation.png')}
                        className="text-primary-600 hover:text-primary-700 text-sm flex items-center transition-colors"
                      >
                        <Download className="h-4 w-4 mr-1" />
                        Download
                      </button>
                    </div>
                    <div className="bg-gray-100 p-4 rounded-lg">
                      <img
                        src={result.perturbation}
                        alt="Perturbation"
                        className="w-full max-w-xs mx-auto rounded border-2 border-yellow-300 shadow-sm"
                      />
                    </div>
                    <p className="text-xs text-gray-600 mt-2 text-center">
                      Actual perturbation is 10× smaller than shown (imperceptible to humans)
                    </p>
                  </div>
                </div>
              </div>

              {/* Educational Note */}
              <div className="card bg-blue-50 border border-blue-200">
                <div className="flex items-start">
                  <Info className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                  <div>
                    <h4 className="font-semibold text-blue-900 mb-1">What This Shows</h4>
                    <p className="text-sm text-blue-800">
                      {result.success ? (
                        <>
                          The attack successfully fooled the model with a perturbation that's barely
                          visible to humans. This demonstrates how neural networks can be vulnerable
                          to carefully crafted inputs, highlighting the importance of adversarial
                          robustness in real-world applications.
                        </>
                      ) : (
                        <>
                          The attack did not change the model's prediction. This could mean the
                          perturbation was too small, or the model is naturally more robust to this
                          particular input. Try increasing epsilon or using PGD with more iterations.
                        </>
                      )}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Empty State */}
          {!result && !loading && !error && (
            <div className="card text-center py-16 bg-gray-50">
              <Zap className="h-16 w-16 mx-auto text-gray-400 mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Ready to Generate
              </h3>
              <p className="text-gray-600 mb-4">
                Upload an image and configure the attack parameters,<br />
                then click "Generate Attack" to see the results
              </p>
              <div className="max-w-md mx-auto text-left bg-white p-4 rounded-lg border border-gray-200">
                <h4 className="text-sm font-semibold text-gray-900 mb-2">Quick Tips:</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Start with epsilon = 0.03 for standard attacks</li>
                  <li>• Use FGSM for fast results</li>
                  <li>• Use PGD for stronger attacks</li>
                  <li>• Higher epsilon = more visible perturbations</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AttackDemo;
