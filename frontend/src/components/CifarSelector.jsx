import { useState, useEffect } from 'react';
import { X, RefreshCw, ImageIcon } from 'lucide-react';
import api from '../services/api';
import Loading from './Loading';
import ErrorMessage from './ErrorMessage';

/**
 * CifarSelector Component
 * 
 * Modal component for selecting images from the CIFAR-10 test dataset.
 * Displays a grid of random sample images with their true labels.
 */
const CifarSelector = ({ isOpen, onClose, onSelect }) => {
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedSample, setSelectedSample] = useState(null);

  // Load samples when modal opens
  useEffect(() => {
    if (isOpen) {
      loadSamples();
    }
  }, [isOpen]);

  const loadSamples = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.getCifarSamples(20);
      setSamples(response.samples);
    } catch (err) {
      setError('Failed to load CIFAR-10 samples. Please ensure the backend is running.');
      console.error('Error loading CIFAR samples:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSelect = () => {
    if (selectedSample) {
      onSelect(selectedSample);
      onClose();
    }
  };

  const handleRefresh = () => {
    setSelectedSample(null);
    loadSamples();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 flex items-center">
              <ImageIcon className="h-6 w-6 mr-2 text-primary-600" />
              Select from CIFAR-10 Test Set
            </h2>
            <p className="text-sm text-gray-600 mt-1">
              Choose an image from the CIFAR-10 dataset to test adversarial attacks
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading && (
            <div className="py-12">
              <Loading message="Loading CIFAR-10 samples..." />
            </div>
          )}

          {error && (
            <ErrorMessage 
              message={error} 
              onRetry={loadSamples}
            />
          )}

          {!loading && !error && samples.length > 0 && (
            <>
              <div className="flex justify-between items-center mb-4">
                <p className="text-sm text-gray-600">
                  {samples.length} random samples loaded
                </p>
                <button
                  onClick={handleRefresh}
                  className="btn-outline text-sm flex items-center"
                >
                  <RefreshCw className="h-4 w-4 mr-1" />
                  Load New Samples
                </button>
              </div>

              <div className="grid grid-cols-4 sm:grid-cols-5 md:grid-cols-6 lg:grid-cols-8 gap-3">
                {samples.map((sample) => (
                  <div
                    key={sample.id}
                    onClick={() => setSelectedSample(sample)}
                    className={`cursor-pointer rounded-lg border-2 transition-all hover:shadow-md ${
                      selectedSample?.id === sample.id
                        ? 'border-primary-600 bg-primary-50 shadow-lg scale-105'
                        : 'border-gray-200 hover:border-primary-300'
                    }`}
                  >
                    <div className="p-2">
                      <img
                        src={sample.image}
                        alt={sample.label}
                        className="w-full h-auto rounded"
                      />
                      <p className="text-xs text-center mt-1 font-medium text-gray-700 truncate">
                        {sample.label}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              {selectedSample && (
                <div className="mt-6 p-4 bg-primary-50 border border-primary-200 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-700">Selected:</p>
                      <p className="text-lg font-bold text-primary-700">
                        {selectedSample.label}
                      </p>
                      <p className="text-xs text-gray-600">
                        Image ID: {selectedSample.id}
                      </p>
                    </div>
                    <img
                      src={selectedSample.image}
                      alt={selectedSample.label}
                      className="w-24 h-24 rounded border-2 border-primary-300"
                    />
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-6 border-t border-gray-200 bg-gray-50">
          <button
            onClick={onClose}
            className="btn-secondary"
          >
            Cancel
          </button>
          <button
            onClick={handleSelect}
            disabled={!selectedSample}
            className="btn-primary"
          >
            Use Selected Image
          </button>
        </div>
      </div>
    </div>
  );
};

export default CifarSelector;
