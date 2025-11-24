import { BookOpen, Code } from 'lucide-react';
import MethodologyExplainer from '../components/MethodologyExplainer';

const About = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="section-title">About the Project</h1>
          <p className="section-subtitle">
            Learn about the methodology and techniques behind adversarial attacks
          </p>
        </div>

        {/* Project Overview */}
        <div className="card mb-8">
          <h2 className="text-2xl font-bold mb-4 flex items-center">
            <BookOpen className="h-6 w-6 mr-2 text-primary-600" />
            Project Overview
          </h2>
          <p className="text-gray-700 mb-4">
            This project demonstrates the vulnerability of deep neural networks to adversarial
            attacks. We implement two state-of-the-art attack methods - FGSM and PGD - on a
            ResNet18 model trained on the CIFAR-10 dataset.
          </p>
          <p className="text-gray-700">
            The goal is to provide an educational tool for understanding how small, carefully
            crafted perturbations can fool neural networks, highlighting the importance of
            robustness in machine learning systems.
          </p>
        </div>

        {/* Interactive Methodology Explainer */}
        <MethodologyExplainer />

        {/* Implementation Section */}
        <div className="card mb-8">
          <h2 className="text-2xl font-bold mb-4 flex items-center">
            <Code className="h-6 w-6 mr-2 text-primary-600" />
            Implementation Details
          </h2>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">Model Architecture</h3>
              <p className="text-gray-700">
                ResNet18 convolutional neural network adapted for CIFAR-10 (32×32 RGB images).
                The model achieves ~91% accuracy on clean test images.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">Dataset</h3>
              <p className="text-gray-700">
                CIFAR-10 consists of 60,000 32×32 color images in 10 classes: airplane,
                automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">Attack Parameters</h3>
              <ul className="list-disc list-inside text-gray-700 space-y-1 ml-4">
                <li>Epsilon (ε): Typically 0.03 for standard benchmark (8/255 in pixel space)</li>
                <li>PGD iterations: 20-40 for strong attacks</li>
                <li>PGD step size (α): ε/4 or ε/10</li>
                <li>Random initialization for PGD</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Evaluation Metrics */}
        <div className="card mb-8">
          <h2 className="text-2xl font-bold mb-4">Evaluation Metrics</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">Attack Success Rate (ASR)</h3>
              <p className="text-gray-700 text-sm">
                Percentage of correctly classified clean examples that are misclassified
                after the attack.
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">Adversarial Accuracy</h3>
              <p className="text-gray-700 text-sm">
                Accuracy of the model on adversarial examples.
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">L∞ Norm</h3>
              <p className="text-gray-700 text-sm">
                Maximum per-pixel perturbation. Should equal epsilon for successful attacks.
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">L2 Norm</h3>
              <p className="text-gray-700 text-sm">
                Euclidean distance of the perturbation vector.
              </p>
            </div>
          </div>
        </div>

        {/* References */}
        <div className="card">
          <h2 className="text-2xl font-bold mb-4">References</h2>
          <div className="space-y-4">
            <div className="border-l-4 border-primary-600 pl-4">
              <p className="text-gray-900 font-medium mb-1">
                Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014)
              </p>
              <p className="text-gray-700 text-sm mb-1">
                Explaining and harnessing adversarial examples
              </p>
              <a
                href="https://arxiv.org/abs/1412.6572"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary-600 hover:text-primary-700 text-sm"
              >
                arXiv:1412.6572
              </a>
            </div>
            <div className="border-l-4 border-primary-600 pl-4">
              <p className="text-gray-900 font-medium mb-1">
                Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017)
              </p>
              <p className="text-gray-700 text-sm mb-1">
                Towards deep learning models resistant to adversarial attacks
              </p>
              <a
                href="https://arxiv.org/abs/1706.06083"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary-600 hover:text-primary-700 text-sm"
              >
                arXiv:1706.06083
              </a>
            </div>
            <div className="border-l-4 border-primary-600 pl-4">
              <p className="text-gray-900 font-medium mb-1">
                He, K., Zhang, X., Ren, S., & Sun, J. (2016)
              </p>
              <p className="text-gray-700 text-sm mb-1">
                Deep residual learning for image recognition
              </p>
              <a
                href="https://arxiv.org/abs/1512.03385"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary-600 hover:text-primary-700 text-sm"
              >
                arXiv:1512.03385
              </a>
            </div>
          </div>
        </div>

        {/* Course Info */}
        <div className="mt-8 p-6 bg-primary-50 rounded-lg text-center">
          <p className="text-gray-700">
            <strong>Course:</strong> CS685 - Advanced Topics in Machine Learning
          </p>
          <p className="text-gray-600 text-sm mt-2">
            This project is part of the coursework for CS685
          </p>
        </div>
      </div>
    </div>
  );
};

export default About;
