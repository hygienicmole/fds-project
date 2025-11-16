import { useState, useEffect } from 'react';
import { TrendingUp, Target, Zap } from 'lucide-react';
import api from '../services/api';
import Loading from '../components/Loading';
import ErrorMessage from '../components/ErrorMessage';

const Gallery = () => {
  const [examples, setExamples] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadExamples();
  }, []);

  const loadExamples = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await api.getExampleResults();
      setExamples(data);
    } catch (err) {
      setError('Failed to load example results. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="section-title">Results Gallery</h1>
          <p className="section-subtitle max-w-3xl mx-auto">
            Pre-computed example attack results demonstrating the effectiveness of different adversarial methods
          </p>
        </div>

        {loading && <Loading message="Loading example results..." />}

        {error && <ErrorMessage message={error} onRetry={loadExamples} />}

        {!loading && !error && examples.length > 0 && (
          <div className="space-y-8">
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="card text-center">
                <TrendingUp className="h-12 w-12 text-primary-600 mx-auto mb-3" />
                <div className="text-3xl font-bold text-gray-900 mb-1">
                  {examples.filter(e => e.attack_success).length}
                </div>
                <div className="text-gray-600">Successful Attacks</div>
              </div>
              <div className="card text-center">
                <Target className="h-12 w-12 text-primary-600 mx-auto mb-3" />
                <div className="text-3xl font-bold text-gray-900 mb-1">
                  {examples.length}
                </div>
                <div className="text-gray-600">Total Examples</div>
              </div>
              <div className="card text-center">
                <Zap className="h-12 w-12 text-primary-600 mx-auto mb-3" />
                <div className="text-3xl font-bold text-gray-900 mb-1">
                  {Math.round((examples.filter(e => e.attack_success).length / examples.length) * 100)}%
                </div>
                <div className="text-gray-600">Success Rate</div>
              </div>
            </div>

            {/* Example Results Table */}
            <div className="card">
              <h2 className="text-2xl font-bold mb-6">Example Attack Results</h2>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        ID
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Attack Type
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Epsilon
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Original Class
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Adversarial Class
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Success
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Confidence
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {examples.map((example) => (
                      <tr key={example.image_id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          #{example.image_id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          <span className="px-2 py-1 bg-primary-100 text-primary-800 rounded-full font-medium">
                            {example.attack_type}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {example.epsilon.toFixed(3)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {example.original_class}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {example.adversarial_class}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                            example.attack_success
                              ? 'bg-red-100 text-red-800'
                              : 'bg-green-100 text-green-800'
                          }`}>
                            {example.attack_success ? 'Success' : 'Failed'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {(example.adversarial_confidence * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {!loading && !error && examples.length === 0 && (
          <div className="card text-center py-12">
            <p className="text-gray-600">
              No example results found. Run the attack evaluation to generate examples.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Gallery;
