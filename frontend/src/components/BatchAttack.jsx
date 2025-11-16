import { useState } from 'react';
import {
  Upload,
  Zap,
  Target,
  Download,
  FileText,
  BarChart3,
  AlertCircle,
  CheckCircle,
  XCircle,
  Clock,
  TrendingUp
} from 'lucide-react';
import api from '../services/api';
import Loading from './Loading';
import ErrorMessage from './ErrorMessage';

const BatchAttack = () => {
  const [files, setFiles] = useState([]);
  const [attackType, setAttackType] = useState('fgsm');
  const [epsilon, setEpsilon] = useState(0.03);
  const [pgdIterations, setPgdIterations] = useState(20);
  const [pgdAlpha, setPgdAlpha] = useState(0.0075);
  const [randomStart, setRandomStart] = useState(true);
  const [exportFormat, setExportFormat] = useState('json');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    if (selectedFiles.length > 50) {
      setError('Maximum 50 files allowed');
      return;
    }
    setFiles(selectedFiles);
    setError(null);
  };

  // Remove a file from the list
  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  // Handle batch attack generation
  const handleBatchAttack = async () => {
    if (files.length === 0) {
      setError('Please select at least one image');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const params = {
        attack_type: attackType,
        epsilon: parseFloat(epsilon),
        pgd_alpha: parseFloat(pgdAlpha),
        pgd_iterations: parseInt(pgdIterations),
        random_start: randomStart,
        export_format: exportFormat
      };

      const response = await api.generateBatchAttack(files, params);
      setResults(response);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate batch attacks. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Export results as JSON
  const exportJSON = () => {
    if (!results) return;

    const dataStr = JSON.stringify(results, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `batch_attack_results_${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Export results as CSV
  const exportCSV = () => {
    if (!results || !results.csv_export) {
      // Generate CSV if not already present
      if (!results || !results.results) return;

      const headers = Object.keys(results.results[0]).join(',');
      const rows = results.results.map(r =>
        Object.values(r).map(v =>
          typeof v === 'string' ? `"${v}"` : v
        ).join(',')
      );
      const csv = [headers, ...rows].join('\n');

      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `batch_attack_results_${Date.now()}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } else {
      const blob = new Blob([results.csv_export], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `batch_attack_results_${Date.now()}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Batch Attack Generation</h2>
        <p className="text-gray-600">
          Process multiple images simultaneously and analyze aggregate attack statistics
        </p>
      </div>

      {/* File Upload Section */}
      <div className="card">
        <h3 className="text-xl font-bold text-gray-900 mb-4">Upload Images</h3>

        <div className="mb-4">
          <label
            htmlFor="file-upload"
            className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition"
          >
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <Upload className="h-12 w-12 text-gray-400 mb-3" />
              <p className="mb-2 text-sm text-gray-500">
                <span className="font-semibold">Click to upload</span> or drag and drop
              </p>
              <p className="text-xs text-gray-500">
                PNG, JPG or JPEG (Max 50 files, 10MB each)
              </p>
            </div>
            <input
              id="file-upload"
              type="file"
              className="hidden"
              multiple
              accept="image/*"
              onChange={handleFileChange}
            />
          </label>
        </div>

        {/* Selected Files List */}
        {files.length > 0 && (
          <div className="space-y-2">
            <p className="text-sm font-semibold text-gray-700">
              Selected Files ({files.length}):
            </p>
            <div className="max-h-48 overflow-y-auto space-y-1">
              {files.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between bg-gray-50 p-2 rounded"
                >
                  <span className="text-sm text-gray-700 truncate">{file.name}</span>
                  <button
                    onClick={() => removeFile(index)}
                    className="text-red-600 hover:text-red-800 text-sm ml-2"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Attack Configuration */}
      <div className="card">
        <h3 className="text-xl font-bold text-gray-900 mb-4">Attack Configuration</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Attack Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Attack Type
            </label>
            <div className="flex space-x-4">
              <button
                onClick={() => setAttackType('fgsm')}
                className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-semibold transition ${
                  attackType === 'fgsm'
                    ? 'bg-orange-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                <Zap className="h-5 w-5" />
                <span>FGSM</span>
              </button>
              <button
                onClick={() => setAttackType('pgd')}
                className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-semibold transition ${
                  attackType === 'pgd'
                    ? 'bg-red-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                <Target className="h-5 w-5" />
                <span>PGD</span>
              </button>
            </div>
          </div>

          {/* Epsilon */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Epsilon (ε): {epsilon.toFixed(3)}
            </label>
            <input
              type="range"
              min="0.001"
              max="0.3"
              step="0.001"
              value={epsilon}
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>0.001</span>
              <span>0.3</span>
            </div>
          </div>

          {/* PGD-specific controls */}
          {attackType === 'pgd' && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Iterations: {pgdIterations}
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

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Step Size (α): {pgdAlpha.toFixed(4)}
                </label>
                <input
                  type="range"
                  min="0.001"
                  max="0.05"
                  step="0.001"
                  value={pgdAlpha}
                  onChange={(e) => setPgdAlpha(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0.001</span>
                  <span>0.05</span>
                </div>
              </div>

              <div className="flex items-center">
                <input
                  id="random-start"
                  type="checkbox"
                  checked={randomStart}
                  onChange={(e) => setRandomStart(e.target.checked)}
                  className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                />
                <label htmlFor="random-start" className="ml-2 text-sm text-gray-700">
                  Use Random Initialization
                </label>
              </div>
            </>
          )}

          {/* Export Format */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Export Format
            </label>
            <select
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="json">JSON</option>
              <option value="csv">CSV</option>
            </select>
          </div>
        </div>

        {/* Generate Button */}
        <button
          onClick={handleBatchAttack}
          disabled={files.length === 0 || loading}
          className={`mt-6 w-full flex items-center justify-center space-x-2 py-3 px-6 rounded-lg font-semibold text-white transition ${
            files.length === 0 || loading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-primary-600 hover:bg-primary-700'
          }`}
        >
          <BarChart3 className="h-5 w-5" />
          <span>{loading ? 'Processing...' : 'Generate Batch Attack'}</span>
        </button>
      </div>

      {/* Loading State */}
      {loading && (
        <Loading message={`Processing ${files.length} images with ${attackType.toUpperCase()}...`} />
      )}

      {/* Error Display */}
      {error && <ErrorMessage message={error} onRetry={() => setError(null)} />}

      {/* Results */}
      {results && (
        <div className="space-y-6">
          {/* Summary Statistics */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-2xl font-bold text-gray-900">Summary Statistics</h3>
              <div className="flex space-x-2">
                <button
                  onClick={exportJSON}
                  className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
                >
                  <Download className="h-4 w-4" />
                  <span>JSON</span>
                </button>
                <button
                  onClick={exportCSV}
                  className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"
                >
                  <FileText className="h-4 w-4" />
                  <span>CSV</span>
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-1">
                  <FileText className="h-5 w-5 text-blue-600" />
                  <span className="text-sm font-medium text-blue-900">Total Images</span>
                </div>
                <div className="text-2xl font-bold text-blue-900">
                  {results.summary.total_images}
                </div>
              </div>

              <div className="bg-red-50 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-1">
                  <XCircle className="h-5 w-5 text-red-600" />
                  <span className="text-sm font-medium text-red-900">Attacked</span>
                </div>
                <div className="text-2xl font-bold text-red-900">
                  {results.summary.successful_attacks}
                </div>
              </div>

              <div className="bg-green-50 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-1">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  <span className="text-sm font-medium text-green-900">Robust</span>
                </div>
                <div className="text-2xl font-bold text-green-900">
                  {results.summary.failed_attacks}
                </div>
              </div>

              <div className="bg-purple-50 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-1">
                  <TrendingUp className="h-5 w-5 text-purple-600" />
                  <span className="text-sm font-medium text-purple-900">ASR</span>
                </div>
                <div className="text-2xl font-bold text-purple-900">
                  {results.summary.attack_success_rate}%
                </div>
              </div>
            </div>

            <div className="mt-6 grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Avg L∞ Norm:</span>
                <span className="ml-2 font-semibold text-gray-900">
                  {results.summary.average_linf_norm.toFixed(6)}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Avg L2 Norm:</span>
                <span className="ml-2 font-semibold text-gray-900">
                  {results.summary.average_l2_norm.toFixed(6)}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Processing Time:</span>
                <span className="ml-2 font-semibold text-gray-900">
                  {results.summary.processing_time_seconds}s
                </span>
              </div>
              <div>
                <span className="text-gray-600">Throughput:</span>
                <span className="ml-2 font-semibold text-gray-900">
                  {results.summary.images_per_second.toFixed(2)} img/s
                </span>
              </div>
            </div>
          </div>

          {/* Class-wise Statistics */}
          {results.class_wise_statistics && Object.keys(results.class_wise_statistics).length > 0 && (
            <div className="card">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Class-wise Robustness</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                        Class
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                        Total
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                        Robust
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                        Vulnerable
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                        Robustness %
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {Object.entries(results.class_wise_statistics).map(([className, stats]) => (
                      <tr key={className} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {className}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                          {stats.total}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600 font-semibold">
                          {stats.robust}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600 font-semibold">
                          {stats.vulnerable}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="w-32 bg-gray-200 rounded-full h-2 mr-2">
                              <div
                                className="bg-green-600 h-2 rounded-full"
                                style={{ width: `${stats.robustness_rate}%` }}
                              ></div>
                            </div>
                            <span className="text-sm text-gray-900 font-semibold">
                              {stats.robustness_rate}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Individual Results */}
          <div className="card">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Individual Results ({results.results.length})
            </h3>
            <div className="overflow-x-auto max-h-96">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      #
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Filename
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Original
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Adversarial
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Success
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      L∞ Norm
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {results.results.map((result) => (
                    <tr key={result.image_id} className="hover:bg-gray-50">
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                        {result.image_id + 1}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-700 truncate max-w-xs">
                        {result.filename}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                        {result.original_class}
                        <span className="text-gray-500 text-xs ml-1">
                          ({(result.original_confidence * 100).toFixed(1)}%)
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                        {result.adversarial_class}
                        <span className="text-gray-500 text-xs ml-1">
                          ({(result.adversarial_confidence * 100).toFixed(1)}%)
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        {result.error ? (
                          <span className="px-2 py-1 text-xs font-semibold rounded-full bg-yellow-100 text-yellow-800">
                            Error
                          </span>
                        ) : result.attack_success ? (
                          <span className="px-2 py-1 text-xs font-semibold rounded-full bg-red-100 text-red-800">
                            Success
                          </span>
                        ) : (
                          <span className="px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">
                            Failed
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                        {result.linf_norm ? result.linf_norm.toFixed(6) : 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchAttack;
