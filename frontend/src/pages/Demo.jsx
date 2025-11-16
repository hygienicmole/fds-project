import { useState } from 'react';
import AttackDemo from '../components/AttackDemo';
import BatchAttack from '../components/BatchAttack';
import RealTimeAttack from '../components/RealTimeAttack';

const Demo = () => {
  const [activeTab, setActiveTab] = useState('single');

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Page Header */}
        <div className="text-center mb-12">
          <h1 className="section-title">Interactive Demo</h1>
          <p className="section-subtitle max-w-3xl mx-auto">
            Generate adversarial examples in real-time and observe how small perturbations
            can fool neural networks
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8 space-x-4">
          <button
            onClick={() => setActiveTab('single')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'single'
                ? 'bg-primary-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            Single Attack
          </button>
          <button
            onClick={() => setActiveTab('realtime')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'realtime'
                ? 'bg-primary-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            Real-Time Visualization
          </button>
          <button
            onClick={() => setActiveTab('batch')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'batch'
                ? 'bg-primary-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            Batch Attack
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'single' && <AttackDemo />}
        {activeTab === 'realtime' && <RealTimeAttack />}
        {activeTab === 'batch' && <BatchAttack />}

        {/* Educational Footer */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-2">What is FGSM?</h3>
            <p className="text-sm text-gray-600">
              Fast Gradient Sign Method is a one-step attack that perturbs the input in the
              direction of the gradient sign. It's fast but less effective than iterative methods.
            </p>
          </div>
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-2">What is PGD?</h3>
            <p className="text-sm text-gray-600">
              Projected Gradient Descent is an iterative attack that applies multiple small steps,
              projecting back to the allowed region. It's slower but much more effective.
            </p>
          </div>
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-2">Why Does This Matter?</h3>
            <p className="text-sm text-gray-600">
              Understanding adversarial attacks is crucial for building robust AI systems,
              especially in security-critical applications like autonomous vehicles and medical diagnosis.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Demo;
