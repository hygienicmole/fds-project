import ResultsDashboard from '../components/ResultsDashboard';

const Gallery = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="section-title">Results Gallery</h1>
          <p className="section-subtitle max-w-3xl mx-auto">
            Comprehensive analysis of adversarial attacks with interactive visualizations and metrics
          </p>
        </div>

        {/* Results Dashboard */}
        <ResultsDashboard />
      </div>
    </div>
  );
};

export default Gallery;
