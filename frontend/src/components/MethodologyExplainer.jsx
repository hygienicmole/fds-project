import { useState } from 'react';
import {
  BookOpen,
  Code2,
  Lightbulb,
  ArrowRight,
  GitBranch,
  Zap,
  Target,
  ChevronDown,
  ChevronUp,
  PlayCircle
} from 'lucide-react';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import python from 'react-syntax-highlighter/dist/esm/languages/hljs/python';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

// Register Python language for syntax highlighting
SyntaxHighlighter.registerLanguage('python', python);

const MethodologyExplainer = () => {
  const [activeTab, setActiveTab] = useState('fgsm');
  const [expandedSections, setExpandedSections] = useState({
    fgsmSteps: true,
    fgsmCode: false,
    fgsmVisual: true,
    pgdSteps: true,
    pgdCode: false,
    pgdVisual: true,
  });
  const [animationStep, setAnimationStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const startAnimation = () => {
    if (isAnimating) return;
    setIsAnimating(true);
    setAnimationStep(0);

    const steps = activeTab === 'fgsm' ? 4 : 7;
    let currentStep = 0;

    const interval = setInterval(() => {
      currentStep++;
      if (currentStep > steps) {
        clearInterval(interval);
        setIsAnimating(false);
        setAnimationStep(0);
      } else {
        setAnimationStep(currentStep);
      }
    }, 1500);
  };

  // FGSM Code Example
  const fgsmCode = `def fgsm_attack(model, image, label, epsilon):
    """
    Fast Gradient Sign Method (FGSM) Attack

    Args:
        model: Neural network model
        image: Input image tensor
        label: True label
        epsilon: Perturbation magnitude (L∞ bound)

    Returns:
        adversarial_image: Perturbed image
    """
    # Require gradients for the input image
    image.requires_grad = True

    # Forward pass
    output = model(image)

    # Calculate loss
    loss = F.cross_entropy(output, label)

    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()

    # Get gradient sign
    gradient_sign = image.grad.sign()

    # Create adversarial example
    adversarial_image = image + epsilon * gradient_sign

    # Clamp to valid pixel range [0, 1]
    adversarial_image = torch.clamp(adversarial_image, 0, 1)

    return adversarial_image`;

  // PGD Code Example
  const pgdCode = `def pgd_attack(model, image, label, epsilon, alpha, iterations):
    """
    Projected Gradient Descent (PGD) Attack

    Args:
        model: Neural network model
        image: Input image tensor
        label: True label
        epsilon: Maximum perturbation (L∞ bound)
        alpha: Step size for each iteration
        iterations: Number of attack iterations

    Returns:
        adversarial_image: Perturbed image
    """
    # Start from random point in epsilon-ball
    perturbation = torch.empty_like(image).uniform_(-epsilon, epsilon)
    adversarial_image = torch.clamp(image + perturbation, 0, 1)

    # Iterative attack
    for i in range(iterations):
        adversarial_image.requires_grad = True

        # Forward pass
        output = model(adversarial_image)

        # Calculate loss
        loss = F.cross_entropy(output, label)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update with gradient sign
        gradient_sign = adversarial_image.grad.sign()
        adversarial_image = adversarial_image + alpha * gradient_sign

        # Project back to epsilon-ball around original image
        perturbation = torch.clamp(
            adversarial_image - image,
            -epsilon,
            epsilon
        )
        adversarial_image = torch.clamp(image + perturbation, 0, 1)
        adversarial_image = adversarial_image.detach()

    return adversarial_image`;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <BookOpen className="h-12 w-12 text-primary-600 mx-auto mb-4" />
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Attack Methodology</h2>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          Understanding the mathematical foundations and implementation of adversarial attacks
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex justify-center space-x-4 border-b border-gray-200">
        <button
          onClick={() => setActiveTab('fgsm')}
          className={`px-6 py-3 font-semibold transition-all ${
            activeTab === 'fgsm'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <div className="flex items-center space-x-2">
            <Zap className="h-5 w-5" />
            <span>FGSM</span>
          </div>
        </button>
        <button
          onClick={() => setActiveTab('pgd')}
          className={`px-6 py-3 font-semibold transition-all ${
            activeTab === 'pgd'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <div className="flex items-center space-x-2">
            <Target className="h-5 w-5" />
            <span>PGD</span>
          </div>
        </button>
        <button
          onClick={() => setActiveTab('comparison')}
          className={`px-6 py-3 font-semibold transition-all ${
            activeTab === 'comparison'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <div className="flex items-center space-x-2">
            <GitBranch className="h-5 w-5" />
            <span>Comparison</span>
          </div>
        </button>
      </div>

      {/* FGSM Tab Content */}
      {activeTab === 'fgsm' && (
        <div className="space-y-6">
          {/* Overview */}
          <div className="card">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Fast Gradient Sign Method (FGSM)
            </h3>
            <p className="text-gray-700 mb-4">
              FGSM is a one-step adversarial attack that perturbs the input in the direction of
              the gradient sign to maximize the loss. It's computationally efficient but less
              effective than iterative methods like PGD.
            </p>
            <div className="bg-blue-50 border-l-4 border-blue-500 p-4">
              <div className="flex items-start">
                <Lightbulb className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                <div>
                  <p className="text-sm font-semibold text-blue-900 mb-1">Key Insight</p>
                  <p className="text-sm text-blue-800">
                    The sign function makes FGSM fast but limits its effectiveness. Each pixel
                    is perturbed by exactly ±ε, regardless of the gradient magnitude.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Mathematical Formulation */}
          <div className="card">
            <h4 className="text-xl font-bold text-gray-900 mb-4">Mathematical Formulation</h4>
            <div className="space-y-4">
              <div>
                <p className="text-gray-700 mb-3">
                  Given an input image <InlineMath math="x" />, true label <InlineMath math="y" />,
                  and model <InlineMath math="f_\theta" />, the adversarial example <InlineMath math="x'" />
                  is computed as:
                </p>
                <div className="bg-gray-50 p-4 rounded-lg overflow-x-auto">
                  <BlockMath math="x' = x + \epsilon \cdot \text{sign}(\nabla_x J(f_\theta(x), y))" />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div className="bg-purple-50 p-4 rounded-lg">
                  <p className="font-semibold text-purple-900 mb-2">Where:</p>
                  <ul className="space-y-2 text-sm text-purple-800">
                    <li><InlineMath math="\epsilon" /> - Maximum perturbation (L∞ norm)</li>
                    <li><InlineMath math="J(\cdot)" /> - Loss function (e.g., cross-entropy)</li>
                    <li><InlineMath math="\nabla_x" /> - Gradient with respect to input</li>
                    <li><InlineMath math="\text{sign}(\cdot)" /> - Sign function (±1)</li>
                  </ul>
                </div>

                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="font-semibold text-green-900 mb-2">Properties:</p>
                  <ul className="space-y-2 text-sm text-green-800">
                    <li>✓ Single gradient computation</li>
                    <li>✓ Very fast (O(1) iterations)</li>
                    <li>✓ Guaranteed L∞ bound of ε</li>
                    <li>✗ Less effective than iterative methods</li>
                  </ul>
                </div>
              </div>

              <div className="mt-4">
                <p className="text-gray-700 mb-2">
                  The constraint ensures bounded perturbation:
                </p>
                <div className="bg-gray-50 p-4 rounded-lg overflow-x-auto">
                  <BlockMath math="\|x' - x\|_\infty \leq \epsilon" />
                </div>
              </div>
            </div>
          </div>

          {/* Step-by-Step Algorithm */}
          <div className="card">
            <button
              onClick={() => toggleSection('fgsmSteps')}
              className="w-full flex items-center justify-between text-left"
            >
              <h4 className="text-xl font-bold text-gray-900">Step-by-Step Algorithm</h4>
              {expandedSections.fgsmSteps ? (
                <ChevronUp className="h-6 w-6 text-gray-500" />
              ) : (
                <ChevronDown className="h-6 w-6 text-gray-500" />
              )}
            </button>

            {expandedSections.fgsmSteps && (
              <div className="mt-4 space-y-3">
                {[
                  {
                    step: 1,
                    title: 'Forward Pass',
                    description: 'Compute model prediction on clean input',
                    math: 'output = f_\\theta(x)',
                    color: 'blue'
                  },
                  {
                    step: 2,
                    title: 'Calculate Loss',
                    description: 'Compute loss between prediction and true label',
                    math: 'L = J(output, y)',
                    color: 'purple'
                  },
                  {
                    step: 3,
                    title: 'Compute Gradient',
                    description: 'Calculate gradient of loss with respect to input',
                    math: 'g = \\nabla_x L',
                    color: 'green'
                  },
                  {
                    step: 4,
                    title: 'Generate Perturbation',
                    description: 'Create adversarial perturbation using gradient sign',
                    math: '\\delta = \\epsilon \\cdot \\text{sign}(g)',
                    color: 'orange'
                  },
                  {
                    step: 5,
                    title: 'Create Adversarial Example',
                    description: 'Add perturbation to original input and clamp',
                    math: "x' = \\text{clip}(x + \\delta, 0, 1)",
                    color: 'red'
                  }
                ].map(({ step, title, description, math, color }) => (
                  <div
                    key={step}
                    className={`flex items-start p-4 rounded-lg border-l-4 transition-all ${
                      isAnimating && animationStep >= step
                        ? `border-${color}-500 bg-${color}-50 scale-105`
                        : `border-gray-300 bg-gray-50`
                    }`}
                  >
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-white ${
                      isAnimating && animationStep >= step
                        ? `bg-${color}-500`
                        : 'bg-gray-400'
                    }`}>
                      {step}
                    </div>
                    <div className="ml-4 flex-1">
                      <h5 className="font-semibold text-gray-900">{title}</h5>
                      <p className="text-sm text-gray-600 mt-1">{description}</p>
                      <div className="mt-2 bg-white p-2 rounded border border-gray-200 overflow-x-auto">
                        <InlineMath math={math} />
                      </div>
                    </div>
                  </div>
                ))}

                <button
                  onClick={startAnimation}
                  disabled={isAnimating}
                  className={`w-full mt-4 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-semibold transition-all ${
                    isAnimating
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-primary-600 text-white hover:bg-primary-700'
                  }`}
                >
                  <PlayCircle className="h-5 w-5" />
                  <span>{isAnimating ? 'Animating...' : 'Animate Algorithm Flow'}</span>
                </button>
              </div>
            )}
          </div>

          {/* Visual Diagram */}
          <div className="card">
            <button
              onClick={() => toggleSection('fgsmVisual')}
              className="w-full flex items-center justify-between text-left"
            >
              <h4 className="text-xl font-bold text-gray-900">Gradient Flow Visualization</h4>
              {expandedSections.fgsmVisual ? (
                <ChevronUp className="h-6 w-6 text-gray-500" />
              ) : (
                <ChevronDown className="h-6 w-6 text-gray-500" />
              )}
            </button>

            {expandedSections.fgsmVisual && (
              <div className="mt-4">
                <div className="bg-gradient-to-br from-blue-50 to-purple-50 p-8 rounded-lg">
                  <div className="flex flex-col md:flex-row items-center justify-between space-y-6 md:space-y-0 md:space-x-4">
                    {/* Input Image */}
                    <div className="flex flex-col items-center">
                      <div className="w-24 h-24 bg-blue-200 rounded-lg border-4 border-blue-400 flex items-center justify-center">
                        <span className="text-2xl font-bold text-blue-700">x</span>
                      </div>
                      <p className="mt-2 text-sm font-semibold text-gray-700">Input Image</p>
                      <p className="text-xs text-gray-500">Original</p>
                    </div>

                    {/* Arrow 1 */}
                    <div className="flex flex-col items-center">
                      <ArrowRight className="h-8 w-8 text-purple-600" />
                      <p className="text-xs text-purple-600 font-semibold mt-1">Forward</p>
                    </div>

                    {/* Model */}
                    <div className="flex flex-col items-center">
                      <div className="w-24 h-24 bg-purple-200 rounded-lg border-4 border-purple-400 flex items-center justify-center">
                        <span className="text-xl font-bold text-purple-700">f<sub>θ</sub></span>
                      </div>
                      <p className="mt-2 text-sm font-semibold text-gray-700">Model</p>
                      <p className="text-xs text-gray-500">Neural Network</p>
                    </div>

                    {/* Arrow 2 */}
                    <div className="flex flex-col items-center">
                      <ArrowRight className="h-8 w-8 text-green-600" />
                      <p className="text-xs text-green-600 font-semibold mt-1">Loss</p>
                    </div>

                    {/* Loss */}
                    <div className="flex flex-col items-center">
                      <div className="w-24 h-24 bg-green-200 rounded-lg border-4 border-green-400 flex items-center justify-center">
                        <span className="text-2xl font-bold text-green-700">J</span>
                      </div>
                      <p className="mt-2 text-sm font-semibold text-gray-700">Loss</p>
                      <p className="text-xs text-gray-500">Cross-Entropy</p>
                    </div>
                  </div>

                  {/* Gradient Flow Back */}
                  <div className="mt-8 flex items-center justify-center">
                    <div className="w-full max-w-2xl bg-orange-100 border-2 border-dashed border-orange-400 rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <div className="w-12 h-12 bg-orange-300 rounded-full flex items-center justify-center">
                            <span className="text-lg font-bold text-orange-800">∇</span>
                          </div>
                          <div>
                            <p className="font-semibold text-orange-900">Gradient Backpropagation</p>
                            <p className="text-sm text-orange-700">Computing ∇<sub>x</sub>J</p>
                          </div>
                        </div>
                        <ArrowRight className="h-6 w-6 text-orange-600 transform rotate-180" />
                      </div>
                    </div>
                  </div>

                  {/* Final Perturbation */}
                  <div className="mt-8 flex flex-col items-center">
                    <div className="w-32 h-32 bg-red-200 rounded-lg border-4 border-red-400 flex flex-col items-center justify-center">
                      <span className="text-2xl font-bold text-red-700">x'</span>
                      <span className="text-xs text-red-600 mt-1">= x + ε·sign(∇)</span>
                    </div>
                    <p className="mt-2 text-sm font-semibold text-gray-700">Adversarial Example</p>
                    <p className="text-xs text-gray-500">Perturbed Input</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Code Implementation */}
          <div className="card">
            <button
              onClick={() => toggleSection('fgsmCode')}
              className="w-full flex items-center justify-between text-left"
            >
              <div className="flex items-center space-x-2">
                <Code2 className="h-6 w-6 text-gray-700" />
                <h4 className="text-xl font-bold text-gray-900">Python Implementation</h4>
              </div>
              {expandedSections.fgsmCode ? (
                <ChevronUp className="h-6 w-6 text-gray-500" />
              ) : (
                <ChevronDown className="h-6 w-6 text-gray-500" />
              )}
            </button>

            {expandedSections.fgsmCode && (
              <div className="mt-4">
                <SyntaxHighlighter
                  language="python"
                  style={atomOneDark}
                  customStyle={{
                    borderRadius: '0.5rem',
                    padding: '1.5rem',
                  }}
                  showLineNumbers
                >
                  {fgsmCode}
                </SyntaxHighlighter>
              </div>
            )}
          </div>
        </div>
      )}

      {/* PGD Tab Content */}
      {activeTab === 'pgd' && (
        <div className="space-y-6">
          {/* Overview */}
          <div className="card">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Projected Gradient Descent (PGD)
            </h3>
            <p className="text-gray-700 mb-4">
              PGD is an iterative adversarial attack that applies multiple small steps in the
              gradient direction, projecting back to the allowed perturbation region after each step.
              It's considered one of the strongest first-order attacks.
            </p>
            <div className="bg-purple-50 border-l-4 border-purple-500 p-4">
              <div className="flex items-start">
                <Lightbulb className="h-5 w-5 text-purple-600 mt-0.5 mr-3 flex-shrink-0" />
                <div>
                  <p className="text-sm font-semibold text-purple-900 mb-1">Key Insight</p>
                  <p className="text-sm text-purple-800">
                    PGD is essentially FGSM applied iteratively with smaller step sizes. The projection
                    step ensures perturbations stay within the ε-ball, making it a constrained optimization.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Mathematical Formulation */}
          <div className="card">
            <h4 className="text-xl font-bold text-gray-900 mb-4">Mathematical Formulation</h4>
            <div className="space-y-4">
              <div>
                <p className="text-gray-700 mb-3">
                  PGD solves the constrained optimization problem:
                </p>
                <div className="bg-gray-50 p-4 rounded-lg overflow-x-auto">
                  <BlockMath math="\max_{\delta : \|\delta\|_\infty \leq \epsilon} J(f_\theta(x + \delta), y)" />
                </div>
              </div>

              <div>
                <p className="text-gray-700 mb-3">
                  The iterative update rule with projection:
                </p>
                <div className="bg-gray-50 p-4 rounded-lg overflow-x-auto space-y-2">
                  <BlockMath math="x^{(t+1)} = \Pi_{x + \mathcal{S}} \left( x^{(t)} + \alpha \cdot \text{sign}(\nabla_x J(f_\theta(x^{(t)}), y)) \right)" />
                  <div className="text-sm text-gray-600 mt-2">
                    <p>where <InlineMath math="\Pi_{x + \mathcal{S}}" /> projects back to the ε-ball around x</p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div className="bg-purple-50 p-4 rounded-lg">
                  <p className="font-semibold text-purple-900 mb-2">Parameters:</p>
                  <ul className="space-y-2 text-sm text-purple-800">
                    <li><InlineMath math="\epsilon" /> - Maximum perturbation (L∞ bound)</li>
                    <li><InlineMath math="\alpha" /> - Step size per iteration</li>
                    <li><InlineMath math="T" /> - Number of iterations</li>
                    <li><InlineMath math="x^{(0)}" /> - Initial point (random or clean)</li>
                  </ul>
                </div>

                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="font-semibold text-green-900 mb-2">Properties:</p>
                  <ul className="space-y-2 text-sm text-green-800">
                    <li>✓ Much more effective than FGSM</li>
                    <li>✓ Stronger attack (higher ASR)</li>
                    <li>✓ Better exploration of loss surface</li>
                    <li>✗ Slower (O(T) iterations)</li>
                  </ul>
                </div>
              </div>

              <div className="mt-4 bg-orange-50 p-4 rounded-lg">
                <p className="font-semibold text-orange-900 mb-2">Typical Hyperparameters:</p>
                <div className="grid grid-cols-3 gap-4 text-sm text-orange-800">
                  <div>
                    <p className="font-semibold">ε = 0.03</p>
                    <p className="text-xs">Max perturbation</p>
                  </div>
                  <div>
                    <p className="font-semibold">α = ε/4</p>
                    <p className="text-xs">Step size</p>
                  </div>
                  <div>
                    <p className="font-semibold">T = 7-40</p>
                    <p className="text-xs">Iterations</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Step-by-Step Algorithm */}
          <div className="card">
            <button
              onClick={() => toggleSection('pgdSteps')}
              className="w-full flex items-center justify-between text-left"
            >
              <h4 className="text-xl font-bold text-gray-900">Step-by-Step Algorithm</h4>
              {expandedSections.pgdSteps ? (
                <ChevronUp className="h-6 w-6 text-gray-500" />
              ) : (
                <ChevronDown className="h-6 w-6 text-gray-500" />
              )}
            </button>

            {expandedSections.pgdSteps && (
              <div className="mt-4 space-y-3">
                {[
                  {
                    step: 1,
                    title: 'Initialize',
                    description: 'Start from random point in ε-ball around clean image',
                    math: 'x^{(0)} = x + \\mathcal{U}(-\\epsilon, \\epsilon)',
                    color: 'blue'
                  },
                  {
                    step: 2,
                    title: 'Forward Pass',
                    description: 'Compute model prediction on current adversarial image',
                    math: 'output = f_\\theta(x^{(t)})',
                    color: 'purple'
                  },
                  {
                    step: 3,
                    title: 'Calculate Loss',
                    description: 'Compute loss between prediction and true label',
                    math: 'L = J(output, y)',
                    color: 'green'
                  },
                  {
                    step: 4,
                    title: 'Compute Gradient',
                    description: 'Calculate gradient of loss with respect to current image',
                    math: 'g = \\nabla_{x^{(t)}} L',
                    color: 'orange'
                  },
                  {
                    step: 5,
                    title: 'Update',
                    description: 'Take step in gradient sign direction',
                    math: 'x^{(t+1)} = x^{(t)} + \\alpha \\cdot \\text{sign}(g)',
                    color: 'red'
                  },
                  {
                    step: 6,
                    title: 'Project to ε-ball',
                    description: 'Ensure perturbation stays within allowed region',
                    math: '\\delta = \\text{clip}(x^{(t+1)} - x, -\\epsilon, \\epsilon)',
                    color: 'pink'
                  },
                  {
                    step: 7,
                    title: 'Clamp & Repeat',
                    description: 'Ensure valid pixel values and repeat steps 2-6',
                    math: 'x^{(t+1)} = \\text{clip}(x + \\delta, 0, 1)',
                    color: 'indigo'
                  }
                ].map(({ step, title, description, math, color }) => (
                  <div
                    key={step}
                    className={`flex items-start p-4 rounded-lg border-l-4 transition-all ${
                      isAnimating && animationStep >= step
                        ? `border-${color}-500 bg-${color}-50 scale-105`
                        : `border-gray-300 bg-gray-50`
                    }`}
                  >
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-white ${
                      isAnimating && animationStep >= step
                        ? `bg-${color}-500`
                        : 'bg-gray-400'
                    }`}>
                      {step}
                    </div>
                    <div className="ml-4 flex-1">
                      <h5 className="font-semibold text-gray-900">{title}</h5>
                      <p className="text-sm text-gray-600 mt-1">{description}</p>
                      <div className="mt-2 bg-white p-2 rounded border border-gray-200 overflow-x-auto">
                        <InlineMath math={math} />
                      </div>
                    </div>
                  </div>
                ))}

                <button
                  onClick={startAnimation}
                  disabled={isAnimating}
                  className={`w-full mt-4 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-semibold transition-all ${
                    isAnimating
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-primary-600 text-white hover:bg-primary-700'
                  }`}
                >
                  <PlayCircle className="h-5 w-5" />
                  <span>{isAnimating ? 'Animating...' : 'Animate Algorithm Flow'}</span>
                </button>
              </div>
            )}
          </div>

          {/* Visual Diagram */}
          <div className="card">
            <button
              onClick={() => toggleSection('pgdVisual')}
              className="w-full flex items-center justify-between text-left"
            >
              <h4 className="text-xl font-bold text-gray-900">Iterative Process Visualization</h4>
              {expandedSections.pgdVisual ? (
                <ChevronUp className="h-6 w-6 text-gray-500" />
              ) : (
                <ChevronDown className="h-6 w-6 text-gray-500" />
              )}
            </button>

            {expandedSections.pgdVisual && (
              <div className="mt-4">
                <div className="bg-gradient-to-br from-purple-50 to-pink-50 p-8 rounded-lg">
                  {/* Iteration Loop */}
                  <div className="flex flex-col items-center space-y-6">
                    {/* Initial */}
                    <div className="flex items-center space-x-4">
                      <div className="flex flex-col items-center">
                        <div className="w-24 h-24 bg-blue-200 rounded-lg border-4 border-blue-400 flex items-center justify-center">
                          <span className="text-xl font-bold text-blue-700">x<sup>(0)</sup></span>
                        </div>
                        <p className="mt-2 text-sm font-semibold text-gray-700">Random Init</p>
                      </div>
                    </div>

                    {/* Iteration Box */}
                    <div className="w-full max-w-3xl border-4 border-dashed border-purple-400 rounded-lg p-6 bg-white">
                      <div className="flex items-center justify-between mb-4">
                        <p className="font-bold text-purple-900 text-lg">Iteration Loop (t = 1 to T)</p>
                        <div className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm font-semibold">
                          Repeat T times
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        {/* Step 1 */}
                        <div className="flex flex-col items-center">
                          <div className="w-20 h-20 bg-purple-200 rounded-lg border-2 border-purple-400 flex items-center justify-center">
                            <span className="text-sm font-bold text-purple-700">Forward</span>
                          </div>
                          <p className="mt-2 text-xs text-center text-gray-600">f<sub>θ</sub>(x<sup>(t)</sup>)</p>
                        </div>

                        {/* Step 2 */}
                        <div className="flex flex-col items-center">
                          <div className="w-20 h-20 bg-green-200 rounded-lg border-2 border-green-400 flex items-center justify-center">
                            <span className="text-sm font-bold text-green-700">Gradient</span>
                          </div>
                          <p className="mt-2 text-xs text-center text-gray-600">∇<sub>x</sub> J</p>
                        </div>

                        {/* Step 3 */}
                        <div className="flex flex-col items-center">
                          <div className="w-20 h-20 bg-orange-200 rounded-lg border-2 border-orange-400 flex items-center justify-center">
                            <span className="text-sm font-bold text-orange-700">Update</span>
                          </div>
                          <p className="mt-2 text-xs text-center text-gray-600">+ α·sign(∇)</p>
                        </div>

                        {/* Step 4 */}
                        <div className="flex flex-col items-center">
                          <div className="w-20 h-20 bg-red-200 rounded-lg border-2 border-red-400 flex items-center justify-center">
                            <span className="text-sm font-bold text-red-700">Project</span>
                          </div>
                          <p className="mt-2 text-xs text-center text-gray-600">clip to ε-ball</p>
                        </div>
                      </div>

                      <div className="mt-4 flex items-center justify-center">
                        <div className="bg-pink-100 text-pink-800 px-4 py-2 rounded-lg text-sm">
                          x<sup>(t+1)</sup> = Π<sub>ε</sub>( x<sup>(t)</sup> + α·sign(∇<sub>x</sub>J) )
                        </div>
                      </div>
                    </div>

                    {/* Final Output */}
                    <div className="flex flex-col items-center">
                      <div className="w-32 h-32 bg-red-200 rounded-lg border-4 border-red-400 flex flex-col items-center justify-center">
                        <span className="text-2xl font-bold text-red-700">x<sup>(T)</sup></span>
                        <span className="text-xs text-red-600 mt-1">Adversarial</span>
                      </div>
                      <p className="mt-2 text-sm font-semibold text-gray-700">Final Result</p>
                      <p className="text-xs text-gray-500">After T iterations</p>
                    </div>
                  </div>

                  {/* Comparison with FGSM */}
                  <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-blue-100 p-4 rounded-lg">
                      <p className="font-semibold text-blue-900 mb-2">FGSM (T=1)</p>
                      <p className="text-sm text-blue-800">One large step of size ε</p>
                      <div className="mt-2 h-2 bg-blue-300 rounded" style={{ width: '100%' }}></div>
                    </div>
                    <div className="bg-purple-100 p-4 rounded-lg">
                      <p className="font-semibold text-purple-900 mb-2">PGD (T=7)</p>
                      <p className="text-sm text-purple-800">Seven smaller steps, each projected</p>
                      <div className="mt-2 flex space-x-1">
                        {[...Array(7)].map((_, i) => (
                          <div key={i} className="h-2 bg-purple-400 rounded flex-1"></div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Code Implementation */}
          <div className="card">
            <button
              onClick={() => toggleSection('pgdCode')}
              className="w-full flex items-center justify-between text-left"
            >
              <div className="flex items-center space-x-2">
                <Code2 className="h-6 w-6 text-gray-700" />
                <h4 className="text-xl font-bold text-gray-900">Python Implementation</h4>
              </div>
              {expandedSections.pgdCode ? (
                <ChevronUp className="h-6 w-6 text-gray-500" />
              ) : (
                <ChevronDown className="h-6 w-6 text-gray-500" />
              )}
            </button>

            {expandedSections.pgdCode && (
              <div className="mt-4">
                <SyntaxHighlighter
                  language="python"
                  style={atomOneDark}
                  customStyle={{
                    borderRadius: '0.5rem',
                    padding: '1.5rem',
                  }}
                  showLineNumbers
                >
                  {pgdCode}
                </SyntaxHighlighter>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Comparison Tab Content */}
      {activeTab === 'comparison' && (
        <div className="space-y-6">
          {/* Comparison Header */}
          <div className="card">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">FGSM vs PGD</h3>
            <p className="text-gray-700">
              A comprehensive comparison of the two most common white-box adversarial attacks
            </p>
          </div>

          {/* Side-by-Side Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* FGSM Card */}
            <div className="card border-l-4 border-orange-500">
              <div className="flex items-center space-x-3 mb-4">
                <Zap className="h-8 w-8 text-orange-600" />
                <h4 className="text-xl font-bold text-gray-900">FGSM</h4>
              </div>

              <div className="space-y-4">
                <div>
                  <p className="font-semibold text-gray-900 mb-1">Algorithm Type</p>
                  <p className="text-sm text-gray-600">Single-step gradient attack</p>
                </div>

                <div>
                  <p className="font-semibold text-gray-900 mb-1">Computational Cost</p>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div className="bg-green-500 h-2 rounded-full" style={{ width: '20%' }}></div>
                    </div>
                    <span className="text-sm text-gray-600">Low</span>
                  </div>
                </div>

                <div>
                  <p className="font-semibold text-gray-900 mb-1">Attack Success Rate</p>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div className="bg-orange-500 h-2 rounded-full" style={{ width: '60%' }}></div>
                    </div>
                    <span className="text-sm text-gray-600">Moderate</span>
                  </div>
                </div>

                <div>
                  <p className="font-semibold text-gray-900 mb-1">Typical Use Cases</p>
                  <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside">
                    <li>Fast adversarial training</li>
                    <li>Quick baseline evaluation</li>
                    <li>Resource-constrained scenarios</li>
                  </ul>
                </div>

                <div className="bg-orange-50 p-3 rounded-lg">
                  <p className="font-semibold text-orange-900 text-sm mb-1">Formula</p>
                  <div className="overflow-x-auto">
                    <InlineMath math="x' = x + \epsilon \cdot \text{sign}(\nabla_x J)" />
                  </div>
                </div>
              </div>
            </div>

            {/* PGD Card */}
            <div className="card border-l-4 border-red-500">
              <div className="flex items-center space-x-3 mb-4">
                <Target className="h-8 w-8 text-red-600" />
                <h4 className="text-xl font-bold text-gray-900">PGD</h4>
              </div>

              <div className="space-y-4">
                <div>
                  <p className="font-semibold text-gray-900 mb-1">Algorithm Type</p>
                  <p className="text-sm text-gray-600">Multi-step iterative attack</p>
                </div>

                <div>
                  <p className="font-semibold text-gray-900 mb-1">Computational Cost</p>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div className="bg-red-500 h-2 rounded-full" style={{ width: '80%' }}></div>
                    </div>
                    <span className="text-sm text-gray-600">High</span>
                  </div>
                </div>

                <div>
                  <p className="font-semibold text-gray-900 mb-1">Attack Success Rate</p>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div className="bg-green-500 h-2 rounded-full" style={{ width: '95%' }}></div>
                    </div>
                    <span className="text-sm text-gray-600">Very High</span>
                  </div>
                </div>

                <div>
                  <p className="font-semibold text-gray-900 mb-1">Typical Use Cases</p>
                  <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside">
                    <li>Robust adversarial training</li>
                    <li>Security evaluation</li>
                    <li>Benchmark comparisons</li>
                  </ul>
                </div>

                <div className="bg-red-50 p-3 rounded-lg">
                  <p className="font-semibold text-red-900 text-sm mb-1">Formula</p>
                  <div className="overflow-x-auto">
                    <InlineMath math="x^{(t+1)} = \Pi_{\epsilon}(x^{(t)} + \alpha \cdot \text{sign}(\nabla_x J))" />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Detailed Comparison Table */}
          <div className="card overflow-x-auto">
            <h4 className="text-xl font-bold text-gray-900 mb-4">Detailed Comparison</h4>
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Aspect
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    FGSM
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    PGD
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {[
                  {
                    aspect: 'Iterations',
                    fgsm: '1 (single step)',
                    pgd: '7-40 (typically 7-20)'
                  },
                  {
                    aspect: 'Step Size',
                    fgsm: 'ε (full budget)',
                    pgd: 'α = ε/4 (small steps)'
                  },
                  {
                    aspect: 'Initialization',
                    fgsm: 'Clean image x',
                    pgd: 'Random in ε-ball'
                  },
                  {
                    aspect: 'Projection',
                    fgsm: 'No projection needed',
                    pgd: 'Project after each step'
                  },
                  {
                    aspect: 'Time Complexity',
                    fgsm: 'O(1) - Very fast',
                    pgd: 'O(T) - T times slower'
                  },
                  {
                    aspect: 'ASR @ ε=0.03',
                    fgsm: '~40-60%',
                    pgd: '~85-95%'
                  },
                  {
                    aspect: 'Gradient Computations',
                    fgsm: '1 backward pass',
                    pgd: 'T backward passes'
                  },
                  {
                    aspect: 'Perturbation Quality',
                    fgsm: 'Greedy, suboptimal',
                    pgd: 'Better exploration'
                  },
                  {
                    aspect: 'Best Use Case',
                    fgsm: 'Fast evaluation, baselines',
                    pgd: 'Security testing, robustness'
                  }
                ].map(({ aspect, fgsm, pgd }) => (
                  <tr key={aspect} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {aspect}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {fgsm}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {pgd}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Key Takeaways */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="card bg-blue-50 border-l-4 border-blue-500">
              <Lightbulb className="h-8 w-8 text-blue-600 mb-3" />
              <h5 className="font-semibold text-blue-900 mb-2">When to use FGSM</h5>
              <p className="text-sm text-blue-800">
                Use FGSM when you need fast adversarial examples for testing, debugging, or
                when computational resources are limited. Good for quick baselines.
              </p>
            </div>

            <div className="card bg-purple-50 border-l-4 border-purple-500">
              <Lightbulb className="h-8 w-8 text-purple-600 mb-3" />
              <h5 className="font-semibold text-purple-900 mb-2">When to use PGD</h5>
              <p className="text-sm text-purple-800">
                Use PGD for rigorous security evaluation, adversarial training, and when you
                need the strongest possible attack within the ε-ball. Gold standard for benchmarks.
              </p>
            </div>

            <div className="card bg-green-50 border-l-4 border-green-500">
              <Lightbulb className="h-8 w-8 text-green-600 mb-3" />
              <h5 className="font-semibold text-green-900 mb-2">The Trade-off</h5>
              <p className="text-sm text-green-800">
                FGSM is ~10-20x faster but achieves 40-50% lower attack success rate. PGD
                is the standard for evaluating adversarial robustness despite higher cost.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MethodologyExplainer;
