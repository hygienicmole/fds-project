import { Link } from 'react-router-dom';
import { Shield, Zap, Eye, BookOpen, ArrowRight } from 'lucide-react';

const Home = () => {
  const features = [
    {
      icon: <Shield className="h-8 w-8" />,
      title: 'FGSM & PGD Attacks',
      description: 'Implementation of state-of-the-art adversarial attacks on deep learning models.',
    },
    {
      icon: <Zap className="h-8 w-8" />,
      title: 'Real-time Generation',
      description: 'Generate adversarial examples instantly with configurable parameters.',
    },
    {
      icon: <Eye className="h-8 w-8" />,
      title: 'Visual Analysis',
      description: 'Comprehensive visualizations of perturbations and attack effectiveness.',
    },
    {
      icon: <BookOpen className="h-8 w-8" />,
      title: 'Educational',
      description: 'Learn about adversarial machine learning through interactive demonstrations.',
    },
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-primary-600 to-primary-800 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center animate-fade-in">
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              Adversarial Machine Learning
            </h1>
            <p className="text-xl md:text-2xl text-primary-100 mb-8 max-w-3xl mx-auto">
              Interactive demonstration of adversarial attacks on deep neural networks
              using FGSM and PGD techniques.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/demo"
                className="btn-primary bg-white text-primary-600 hover:bg-gray-100 inline-flex items-center justify-center"
              >
                Try Interactive Demo
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
              <Link
                to="/about"
                className="btn-outline border-white text-white hover:bg-white/10 inline-flex items-center justify-center"
              >
                Learn More
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="section-title">Key Features</h2>
            <p className="section-subtitle max-w-2xl mx-auto">
              Explore the capabilities of our adversarial attack demonstration system
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div
                key={index}
                className="card text-center animate-slide-up"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-100 text-primary-600 rounded-lg mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="section-title">How It Works</h2>
            <p className="section-subtitle max-w-2xl mx-auto">
              Simple steps to generate and analyze adversarial examples
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="card text-center">
              <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-600 text-white rounded-full mb-4 text-xl font-bold">
                1
              </div>
              <h3 className="text-xl font-semibold mb-2">Upload Image</h3>
              <p className="text-gray-600">
                Upload a CIFAR-10 image or use our sample images to get started.
              </p>
            </div>

            <div className="card text-center">
              <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-600 text-white rounded-full mb-4 text-xl font-bold">
                2
              </div>
              <h3 className="text-xl font-semibold mb-2">Configure Attack</h3>
              <p className="text-gray-600">
                Choose attack type (FGSM/PGD) and adjust parameters like epsilon.
              </p>
            </div>

            <div className="card text-center">
              <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-600 text-white rounded-full mb-4 text-xl font-bold">
                3
              </div>
              <h3 className="text-xl font-semibold mb-2">Analyze Results</h3>
              <p className="text-gray-600">
                View original and adversarial images with detailed analysis.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary-600 text-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Ready to Explore?
          </h2>
          <p className="text-xl text-primary-100 mb-8">
            Start generating adversarial examples and understand model vulnerabilities
          </p>
          <Link
            to="/demo"
            className="btn-primary bg-white text-primary-600 hover:bg-gray-100 inline-flex items-center"
          >
            Launch Demo
            <ArrowRight className="ml-2 h-5 w-5" />
          </Link>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold text-primary-600 mb-2">2</div>
              <div className="text-gray-600">Attack Methods</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-primary-600 mb-2">10</div>
              <div className="text-gray-600">CIFAR-10 Classes</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-primary-600 mb-2">91%</div>
              <div className="text-gray-600">Baseline Accuracy</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-primary-600 mb-2">89%</div>
              <div className="text-gray-600">Max Attack Success</div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
