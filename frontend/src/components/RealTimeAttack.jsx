import { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Play,
  Pause,
  RotateCcw,
  SkipBack,
  SkipForward,
  Zap,
  TrendingUp
} from 'lucide-react';
import api from '../services/api';
import Loading from './Loading';
import ErrorMessage from './ErrorMessage';
import CifarSelector from './CifarSelector';

const RealTimeAttack = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [epsilon, setEpsilon] = useState(0.03);
  const [pgdIterations, setPgdIterations] = useState(20);
  const [pgdAlpha, setPgdAlpha] = useState(0.0075);
  const [randomStart, setRandomStart] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [attackData, setAttackData] = useState(null);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(500); // ms per iteration
  const [isSelectorOpen, setIsSelectorOpen] = useState(false);
  const [cleanPrediction, setCleanPrediction] = useState(null);
  const intervalRef = useRef(null);

  // Fetch prediction for clean image
  const fetchPrediction = async (imageFile) => {
    try {
      const response = await api.uploadImage(imageFile);
      setCleanPrediction(response.prediction);
    } catch (err) {
      console.error('Error fetching prediction:', err);
      // Don't set main error state here to avoid blocking the UI, just log it
    }
  };

  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setError(null);
    setAttackData(null);
    setCleanPrediction(null);
    setCurrentIteration(0);

    // Generate preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);

    // Fetch prediction immediately
    fetchPrediction(selectedFile);
  };

  // Generate attack
  const handleGenerateAttack = async () => {
    if (!file) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    setError(null);
    setAttackData(null);
    setCurrentIteration(0);
    setIsPlaying(false);

    try {
      const params = {
        epsilon: parseFloat(epsilon),
        pgd_alpha: parseFloat(pgdAlpha),
        pgd_iterations: parseInt(pgdIterations),
        random_start: randomStart,
      };

      const response = await api.generateAttackIterations(file, params);
      setAttackData(response);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate attack. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Play/Pause animation
  const togglePlayPause = () => {
    if (!attackData) return;

    if (isPlaying) {
      // Pause
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setIsPlaying(false);
    } else {
      // Play
      setIsPlaying(true);
    }
  };

  // Handle playback
  useEffect(() => {
    if (isPlaying && attackData) {
      intervalRef.current = setInterval(() => {
        setCurrentIteration((prev) => {
          if (prev >= attackData.iterations.length - 1) {
            // Reached the end
            clearInterval(intervalRef.current);
            intervalRef.current = null;
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, playbackSpeed);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, attackData, playbackSpeed]);

  // Reset to beginning
  const handleReset = () => {
    setCurrentIteration(0);
    setIsPlaying(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  // Step forward/backward
  const stepBackward = () => {
    setCurrentIteration((prev) => Math.max(0, prev - 1));
    setIsPlaying(false);
  };

  const stepForward = () => {
    if (!attackData) return;
    setCurrentIteration((prev) => Math.min(attackData.iterations.length - 1, prev + 1));
    setIsPlaying(false);
  };

  // Get current iteration data
  const getCurrentIterationData = () => {
    if (!attackData || !attackData.iterations[currentIteration]) {
      return null;
    }
    return attackData.iterations[currentIteration];
  };

  const currentData = getCurrentIterationData();

  // Handle CIFAR selection
  const handleCifarSelect = async (sample) => {
    try {
      // Convert base64/URL to file
      const res = await fetch(sample.image);
      const blob = await res.blob();
      const file = new File([blob], `cifar_${sample.label}_${sample.id}.png`, { type: 'image/png' });

      setFile(file);
      setPreview(sample.image);
      setError(null);
      setAttackData(null);
      setCleanPrediction(null);
      setCurrentIteration(0);

      // Fetch prediction immediately
      fetchPrediction(file);
    } catch (err) {
      console.error('Error processing CIFAR image:', err);
      setError('Failed to process selected image');
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Real-Time Attack Visualization</h2>
        <p className="text-gray-600">
          Watch PGD attack unfold iteration by iteration with play/pause controls
        </p>
      </div>

      {/* Upload and Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* File Upload */}
        <div className="card">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold text-gray-900">Upload Image</h3>
            <button
              onClick={() => setIsSelectorOpen(true)}
              className="text-sm text-primary-600 hover:text-primary-700 font-medium"
            >
              Select from CIFAR-10
            </button>
          </div>

          <label
            htmlFor="file-upload"
            className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition relative overflow-hidden"
          >
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              {preview ? (
                <img src={preview} alt="Preview" className="h-40 object-contain" />
              ) : (
                <>
                  <Upload className="h-12 w-12 text-gray-400 mb-3" />
                  <p className="mb-2 text-sm text-gray-500">
                    <span className="font-semibold">Click to upload</span>
                  </p>
                  <p className="text-xs text-gray-500">PNG, JPG (max 10MB)</p>
                </>
              )}
            </div>
            <input
              id="file-upload"
              type="file"
              className="hidden"
              accept="image/*"
              onChange={handleFileChange}
            />
          </label>

          {/* Clean Prediction Display */}
          {cleanPrediction && !attackData && (
            <div className="mt-4 p-4 bg-blue-50 border border-blue-100 rounded-lg animate-fade-in">
              <h4 className="text-sm font-semibold text-blue-900 mb-2">Initial Prediction</h4>
              <div className="flex justify-between items-center">
                <div>
                  <span className="text-2xl font-bold text-blue-700 capitalize">
                    {cleanPrediction.predicted_label}
                  </span>
                  <p className="text-xs text-blue-600">
                    Confidence: {(cleanPrediction.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="h-10 w-10 rounded-full bg-blue-200 flex items-center justify-center">
                  <TrendingUp className="h-6 w-6 text-blue-700" />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Attack Configuration */}
        <div className="card">
          <h3 className="text-xl font-bold text-gray-900 mb-4">PGD Configuration</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Epsilon (ε): {epsilon.toFixed(3)}
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
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Iterations: {pgdIterations}
              </label>
              <input
                type="range"
                min="5"
                max="40"
                step="1"
                value={pgdIterations}
                onChange={(e) => setPgdIterations(parseInt(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Step Size (α): {pgdAlpha.toFixed(4)}
              </label>
              <input
                type="range"
                min="0.001"
                max="0.03"
                step="0.0001"
                value={pgdAlpha}
                onChange={(e) => setPgdAlpha(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div className="flex items-center">
              <input
                id="random-start"
                type="checkbox"
                checked={randomStart}
                onChange={(e) => setRandomStart(e.target.checked)}
                className="w-4 h-4 text-primary-600 border-gray-300 rounded"
              />
              <label htmlFor="random-start" className="ml-2 text-sm text-gray-700">
                Random Initialization
              </label>
            </div>

            <button
              onClick={handleGenerateAttack}
              disabled={!file || loading}
              className={`w-full flex items-center justify-center space-x-2 py-3 px-6 rounded-lg font-semibold text-white transition ${!file || loading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-red-600 hover:bg-red-700'
                }`}
            >
              <Zap className="h-5 w-5" />
              <span>{loading ? 'Generating...' : 'Generate Attack'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Loading */}
      {loading && <Loading message="Generating PGD attack iterations..." />}

      {/* Error */}
      {error && <ErrorMessage message={error} onRetry={() => setError(null)} />}

      {/* Attack Visualization */}
      {attackData && currentData && (
        <div className="space-y-6">
          {/* Playback Controls */}
          <div className="card">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex items-center space-x-4">
                <button
                  onClick={handleReset}
                  className="p-2 bg-gray-200 rounded-lg hover:bg-gray-300 transition"
                  title="Reset to beginning"
                >
                  <RotateCcw className="h-5 w-5 text-gray-700" />
                </button>

                <button
                  onClick={stepBackward}
                  className="p-2 bg-gray-200 rounded-lg hover:bg-gray-300 transition"
                  title="Step backward"
                  disabled={currentIteration === 0}
                >
                  <SkipBack className="h-5 w-5 text-gray-700" />
                </button>

                <button
                  onClick={togglePlayPause}
                  className="p-3 bg-primary-600 rounded-lg hover:bg-primary-700 transition"
                  title={isPlaying ? 'Pause' : 'Play'}
                >
                  {isPlaying ? (
                    <Pause className="h-6 w-6 text-white" />
                  ) : (
                    <Play className="h-6 w-6 text-white" />
                  )}
                </button>

                <button
                  onClick={stepForward}
                  className="p-2 bg-gray-200 rounded-lg hover:bg-gray-300 transition"
                  title="Step forward"
                  disabled={currentIteration >= attackData.iterations.length - 1}
                >
                  <SkipForward className="h-5 w-5 text-gray-700" />
                </button>
              </div>

              <div className="flex items-center space-x-4">
                <span className="text-sm font-medium text-gray-700">Speed:</span>
                <select
                  value={playbackSpeed}
                  onChange={(e) => setPlaybackSpeed(parseInt(e.target.value))}
                  className="px-3 py-2 border border-gray-300 rounded-lg"
                >
                  <option value="200">Fast (0.2s)</option>
                  <option value="500">Normal (0.5s)</option>
                  <option value="1000">Slow (1s)</option>
                  <option value="2000">Very Slow (2s)</option>
                </select>
              </div>

              <div className="text-sm font-semibold text-gray-900">
                Iteration: {currentData.iteration} / {attackData.total_iterations}
              </div>
            </div>

            {/* Progress Bar */}
            <div className="mt-4">
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className={`h-3 rounded-full transition-all ${currentData.is_adversarial ? 'bg-red-600' : 'bg-blue-600'
                    }`}
                  style={{
                    width: `${(currentData.iteration / attackData.total_iterations) * 100}%`
                  }}
                ></div>
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Start</span>
                <span>End</span>
              </div>
            </div>
          </div>

          {/* Current Iteration Display */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Image */}
            <div className="card text-center">
              <h4 className="font-semibold text-gray-900 mb-3">
                Current Image (Iteration {currentData.iteration})
              </h4>
              <img
                src={currentData.image.startsWith('data:') ? currentData.image : `data:image/png;base64,${currentData.image}`}
                alt="Current iteration"
                className="w-48 h-48 mx-auto object-contain border-2 border-gray-300 rounded-lg"
              />
            </div>

            {/* Prediction */}
            <div className="card">
              <h4 className="font-semibold text-gray-900 mb-4">Prediction</h4>
              <div className="space-y-4">
                <div>
                  <span className="text-sm text-gray-600">Class:</span>
                  <div className={`text-2xl font-bold mt-1 ${currentData.is_adversarial ? 'text-red-600' : 'text-green-600'
                    }`}>
                    {currentData.predicted_class}
                  </div>
                </div>

                <div>
                  <span className="text-sm text-gray-600">Confidence:</span>
                  <div className="text-2xl font-bold text-gray-900 mt-1">
                    {(currentData.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full"
                      style={{ width: `${currentData.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>

                <div>
                  <span className="text-sm text-gray-600">Status:</span>
                  <div className="mt-1">
                    {currentData.is_adversarial ? (
                      <span className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-semibold">
                        Adversarial!
                      </span>
                    ) : (
                      <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-semibold">
                        Clean
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Statistics */}
            <div className="card">
              <h4 className="font-semibold text-gray-900 mb-4">Attack Statistics</h4>
              <div className="space-y-4">
                <div>
                  <span className="text-sm text-gray-600">Original Class:</span>
                  <div className="text-lg font-semibold text-gray-900 mt-1">
                    {attackData.original_class}
                  </div>
                </div>

                <div>
                  <span className="text-sm text-gray-600">Perturbation (L∞):</span>
                  <div className="text-lg font-semibold text-gray-900 mt-1">
                    {currentData.perturbation_linf.toFixed(6)}
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-2 overflow-hidden">
                    <div
                      className="bg-orange-600 h-2 rounded-full transition-all"
                      style={{ width: `${Math.min((currentData.perturbation_linf / epsilon) * 100, 100)}%` }}
                    ></div>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {((currentData.perturbation_linf / epsilon) * 100).toFixed(1)}% of ε={epsilon.toFixed(3)}
                  </div>
                </div>

                <div>
                  <span className="text-sm text-gray-600">Final Success:</span>
                  <div className="mt-1">
                    {attackData.final_success ? (
                      <span className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-semibold">
                        Attack Succeeded
                      </span>
                    ) : (
                      <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-semibold">
                        Attack Failed
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Iteration Timeline */}
          <div className="card">
            <h4 className="font-semibold text-gray-900 mb-4">Attack Timeline</h4>
            <div className="overflow-x-auto">
              <div className="flex space-x-2 min-w-max pb-2">
                {attackData.iterations.map((iter) => (
                  <button
                    key={iter.iteration}
                    onClick={() => {
                      setCurrentIteration(iter.iteration);
                      setIsPlaying(false);
                    }}
                    className={`flex-shrink-0 p-2 rounded-lg border-2 transition ${iter.iteration === currentIteration
                      ? 'border-primary-600 bg-primary-50'
                      : iter.is_adversarial
                        ? 'border-red-300 bg-red-50'
                        : 'border-gray-300 bg-gray-50'
                      }`}
                    title={`Iteration ${iter.iteration}: ${iter.predicted_class} (${(iter.confidence * 100).toFixed(1)}%)`}
                  >
                    <img
                      src={iter.image.startsWith('data:') ? iter.image : `data:image/png;base64,${iter.image}`}
                      alt={`Iteration ${iter.iteration}`}
                      className="w-16 h-16 object-contain"
                    />
                    <div className="text-xs text-center mt-1 font-semibold">
                      {iter.iteration}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* CIFAR Selector Modal */}
      <CifarSelector
        isOpen={isSelectorOpen}
        onClose={() => setIsSelectorOpen(false)}
        onSelect={handleCifarSelect}
      />
    </div>
  );
};

export default RealTimeAttack;
