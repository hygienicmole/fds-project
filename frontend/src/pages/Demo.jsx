import AttackDemo from '../components/AttackDemo';

const Demo = () => {
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

        {/* Main Demo Component */}
        <AttackDemo />

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
