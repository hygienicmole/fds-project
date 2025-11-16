import { useState } from 'react';
import { Upload, Zap, Download, Info } from 'lucide-react';
import api, { imageUtils, CIFAR10_CLASSES } from '../services/api';
import { formatConfidence, getConfidenceColor } from '../utils/helpers';
import Loading from '../components/Loading';
import ErrorMessage from '../components/ErrorMessage';

const Demo = () => {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [attackType, setAttackType] = useState('fgsm');
  const [epsilon, setEpsilon] = useState(0.03);
  const [pgdIterations, setPgdIterations] = useState(20);
  const [targeted, setTargeted] = useState(false);
  const [targetClass, setTargetClass] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

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
    setError(null);
    setResult(null);
  };

  const handleGenerateAttack = async () => {
    if (!file) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const params = {
        attack_type: attackType,
        epsilon: parseFloat(epsilon),
        targeted,
        pgd_iterations: parseInt(pgdIterations),
        random_start: true,
      };

      if (targeted) {
        params.target_class = parseInt(targetClass);
      }

      const response = await api.generateAttack(file, params);
      setResult(response);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate attack. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (imageData, filename) => {
    imageUtils.downloadImage(imageData, filename);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="section-title">Interactive Demo</h1>
          <p className="section-subtitle max-w-3xl mx-auto">
            Generate adversarial examples in real-time and observe how small perturbations
            can fool neural networks
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel - Controls */}
          <div className="space-y-6">
            {/* Upload Section */}
            <div className="card">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <Upload className="h-5 w-5 mr-2 text-primary-600" />
                Upload Image
              </h2>

              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary-500 transition-colors">
                <input
                  type="file"
                  accept="image/jpeg,image/jpg,image/png"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600 mb-2">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-sm text-gray-500">PNG or JPEG (max 10MB)</p>
                </label>
              </div>

              {previewUrl && (
                <div className="mt-4">
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="w-full h-48 object-contain bg-gray-100 rounded-lg"
                  />
                  <p className="text-sm text-gray-600 mt-2 text-center">
                    {file?.name}
                  </p>
                </div>
              )}
            </div>

            {/* Attack Configuration */}
            <div className="card">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <Zap className="h-5 w-5 mr-2 text-primary-600" />
                Attack Configuration
              </h2>

              <div className="space-y-4">
                {/* Attack Type */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Attack Type
                  </label>
                  <select
                    value={attackType}
                    onChange={(e) => setAttackType(e.target.value)}
                    className="input-field"
                  >
                    <option value="fgsm">FGSM (Fast Gradient Sign Method)</option>
                    <option value="pgd">PGD (Projected Gradient Descent)</option>
                  </select>
                </div>

                {/* Epsilon */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Epsilon (ε): {epsilon}
                  </label>
                  <input
                    type="range"
                    min="0.001"
                    max="0.1"
                    step="0.001"
                    value={epsilon}
                    onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>0.001</span>
                    <span>0.1</span>
                  </div>
                </div>

                {/* PGD Iterations */}
                {attackType === 'pgd' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      PGD Iterations: {pgdIterations}
                    </label>
                    <input
                      type="range"
                      min="5"
                      max="100"
                      step="5"
                      value={pgdIterations}
                      onChange={(e) => setPgdIterations(parseInt(e.target.value))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>5</span>
                      <span>100</span>
                    </div>
                  </div>
                )}

                {/* Targeted Attack */}
                <div>
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={targeted}
                      onChange={(e) => setTargeted(e.target.checked)}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="text-sm font-medium text-gray-700">
                      Targeted Attack
                    </span>
                  </label>
                </div>

                {/* Target Class */}
                {targeted && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Target Class
                    </label>
                    <select
                      value={targetClass}
                      onChange={(e) => setTargetClass(parseInt(e.target.value))}
                      className="input-field"
                    >
                      {CIFAR10_CLASSES.map((className, index) => (
                        <option key={index} value={index}>
                          {index}: {className}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>

              {/* Generate Button */}
              <button
                onClick={handleGenerateAttack}
                disabled={!file || loading}
                className="btn-primary w-full mt-6"
              >
                {loading ? 'Generating...' : 'Generate Adversarial Example'}
              </button>
            </div>
          </div>

          {/* Right Panel - Results */}
          <div>
            {error && <ErrorMessage message={error} onRetry={() => setError(null)} />}

            {loading && <Loading message="Generating adversarial example..." />}

            {result && !loading && (
              <div className="space-y-6">
                {/* Attack Summary */}
                <div className="card">
                  <h2 className="text-xl font-semibold mb-4">Attack Results</h2>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Attack Success:</span>
                      <span className={`font-semibold ${result.success ? 'text-red-600' : 'text-green-600'}`}>
                        {result.success ? 'Yes' : 'No'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">L∞ Norm:</span>
                      <span className="font-semibold">
                        {result.perturbation_stats.linf_norm.toFixed(6)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">L2 Norm:</span>
                      <span className="font-semibold">
                        {result.perturbation_stats.l2_norm.toFixed(4)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Predictions */}
                <div className="card">
                  <h3 className="font-semibold mb-3">Predictions</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Original</div>
                      <div className={`font-semibold ${getConfidenceColor(result.original_prediction.confidence)}`}>
                        {result.original_prediction.predicted_label}
                      </div>
                      <div className="text-sm text-gray-600">
                        {formatConfidence(result.original_prediction.confidence)}
                      </div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Adversarial</div>
                      <div className={`font-semibold ${getConfidenceColor(result.adversarial_prediction.confidence)}`}>
                        {result.adversarial_prediction.predicted_label}
                      </div>
                      <div className="text-sm text-gray-600">
                        {formatConfidence(result.adversarial_prediction.confidence)}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Images */}
                <div className="card">
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <h4 className="font-medium">Original Image</h4>
                        <button
                          onClick={() => handleDownload(result.original_image, 'original.png')}
                          className="text-primary-600 hover:text-primary-700 text-sm flex items-center"
                        >
                          <Download className="h-4 w-4 mr-1" />
                          Download
                        </button>
                      </div>
                      <img
                        src={result.original_image}
                        alt="Original"
                        className="w-full rounded-lg border border-gray-200"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <h4 className="font-medium">Adversarial Image</h4>
                        <button
                          onClick={() => handleDownload(result.adversarial_image, 'adversarial.png')}
                          className="text-primary-600 hover:text-primary-700 text-sm flex items-center"
                        >
                          <Download className="h-4 w-4 mr-1" />
                          Download
                        </button>
                      </div>
                      <img
                        src={result.adversarial_image}
                        alt="Adversarial"
                        className="w-full rounded-lg border border-gray-200"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <h4 className="font-medium">Perturbation (10× amplified)</h4>
                        <button
                          onClick={() => handleDownload(result.perturbation, 'perturbation.png')}
                          className="text-primary-600 hover:text-primary-700 text-sm flex items-center"
                        >
                          <Download className="h-4 w-4 mr-1" />
                          Download
                        </button>
                      </div>
                      <img
                        src={result.perturbation}
                        alt="Perturbation"
                        className="w-full rounded-lg border border-gray-200"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {!result && !loading && !error && (
              <div className="card text-center py-12">
                <Info className="h-16 w-16 mx-auto text-gray-400 mb-4" />
                <p className="text-gray-600">
                  Upload an image and configure attack parameters to get started
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Demo;
