# Adversarial ML Frontend

Modern React frontend application for interactive adversarial machine learning demonstrations.

## Features

- **Interactive Demo**: Generate adversarial examples in real-time with configurable parameters
- **Results Gallery**: Browse pre-computed attack results and statistics
- **Educational Content**: Learn about FGSM and PGD attack methodologies
- **Modern UI**: Built with React, Vite, and Tailwind CSS
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Tech Stack

- **React 18**: Modern React with hooks
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework
- **React Router**: Client-side routing
- **Axios**: HTTP client for API calls
- **Lucide React**: Icon library

## Quick Start

### Prerequisites

- Node.js 16+ and npm/yarn
- Running FastAPI backend (see `../app/backend.py`)

### Installation

```bash
# Install dependencies
npm install

# or with yarn
yarn install
```

### Development

```bash
# Start development server
npm run dev

# Server will run on http://localhost:3000
```

### Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/       # Reusable React components
│   │   ├── Navbar.jsx    # Navigation bar
│   │   ├── Footer.jsx    # Footer with links
│   │   ├── Loading.jsx   # Loading spinner
│   │   ├── ErrorMessage.jsx  # Error display
│   │   └── AttackDemo.jsx    # Main interactive demo component
│   ├── pages/            # Page components
│   │   ├── Home.jsx      # Landing page
│   │   ├── Demo.jsx      # Interactive demo page
│   │   ├── Gallery.jsx   # Results gallery
│   │   └── About.jsx     # Methodology page
│   ├── services/         # API services
│   │   └── api.js        # Axios API client
│   ├── utils/            # Utility functions
│   │   └── helpers.js    # Helper functions
│   ├── App.jsx           # Main app component
│   ├── main.jsx          # App entry point
│   └── index.css         # Global styles
├── public/               # Static assets
├── index.html            # HTML template
├── vite.config.js        # Vite configuration
├── tailwind.config.js    # Tailwind configuration
├── postcss.config.js     # PostCSS configuration
└── package.json          # Dependencies
```

## Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
```

### API Proxy

The Vite development server is configured to proxy API requests to the backend:

```javascript
// vite.config.js
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, '')
    }
  }
}
```

## Pages

### Home (`/`)

Landing page with:
- Project overview
- Key features
- How it works
- Call-to-action

### Interactive Demo (`/demo`)

Main demonstration page with:
- Image upload
- Attack configuration (FGSM/PGD)
- Parameter controls (epsilon, iterations, etc.)
- Real-time attack generation
- Results display with images and statistics

### Results Gallery (`/gallery`)

Pre-computed results with:
- Attack statistics
- Success rates
- Detailed results table
- Visual summaries

### About (`/about`)

Educational content covering:
- Project overview
- FGSM methodology
- PGD methodology
- Implementation details
- Evaluation metrics
- References

## Components

### Navbar

Responsive navigation bar with:
- Brand logo
- Navigation links
- Mobile menu

### Footer

Footer with:
- Project information
- Quick links
- Resources and references
- Social links

### Loading

Loading spinner with customizable message

### ErrorMessage

Error display with:
- Error icon
- Error message
- Optional retry button

### AttackDemo

Comprehensive interactive demo component featuring:

**Image Handling:**
- File upload with drag-and-drop
- File validation (type, size)
- Image preview
- Sample selection (placeholder for CIFAR-10 test set)

**Attack Configuration:**
- Attack type selector (FGSM/PGD)
- Epsilon slider with visual feedback (0.001-0.3)
- PGD-specific controls:
  - Iterations slider (5-100)
  - Step size (alpha) slider
  - Random initialization toggle
- Targeted attack option with class selection

**Results Display:**
- Attack success indicator
- Side-by-side prediction comparison
- Perturbation statistics (L0, L1, L2, L∞ norms)
- Visual comparison:
  - Original image
  - Adversarial image
  - Amplified perturbation visualization
- Download functionality for all images
- Educational tooltips and information

**UI Features:**
- Collapsible info panel
- Loading states with messages
- Error handling with retry
- Responsive design
- Clean, intuitive layout
- Real-time parameter updates
- Visual feedback for all interactions

## API Service

The `src/services/api.js` module provides:

```javascript
// Get model information
const modelInfo = await api.getModelInfo();

// Upload and classify image
const result = await api.uploadImage(file);

// Generate adversarial attack
const attackResult = await api.generateAttack(file, {
  attack_type: 'pgd',
  epsilon: 0.03,
  pgd_iterations: 20
});

// Get example results
const examples = await api.getExampleResults();

// Health check
const health = await api.healthCheck();
```

## Utilities

### Image Utils

```javascript
// Validate image file
const validation = imageUtils.validateImageFile(file);

// Download base64 image
imageUtils.downloadImage(base64Data, 'adversarial.png');

// Convert base64 to blob
const blob = imageUtils.base64ToBlob(base64Data);
```

### Helper Functions

```javascript
// Format confidence score
formatConfidence(0.9234); // "92.34%"

// Get confidence color class
getConfidenceColor(0.92); // "text-green-600"

// Format file size
formatFileSize(1024000); // "1000 KB"

// Validate epsilon
isValidEpsilon(0.03); // true
```

## Styling

### Tailwind CSS

Custom theme configuration in `tailwind.config.js`:

- Custom primary and secondary colors
- Custom animations (fade-in, slide-up)
- Extended color palette

### Custom CSS Classes

```css
/* Buttons */
.btn-primary
.btn-secondary
.btn-outline

/* Cards */
.card

/* Form inputs */
.input-field

/* Headings */
.section-title
.section-subtitle

/* Gradients */
.text-gradient
.bg-gradient-primary
```

## Development Tips

### Hot Reload

Vite provides instant hot module replacement (HMR) during development.

### Code Organization

- Keep components small and focused
- Use custom hooks for complex logic
- Separate API calls into service modules
- Use utilities for reusable functions

### State Management

This project uses React's built-in state management (useState, useEffect).
For larger applications, consider adding Redux or Zustand.

### Error Handling

All API calls include comprehensive error handling with user-friendly messages.

## Performance

### Optimization

- Lazy loading for images
- Code splitting with React Router
- Vite's optimized build process
- Efficient re-renders with React hooks

### Build Output

Production build includes:
- Minified JavaScript
- Optimized CSS
- Tree-shaking for smaller bundles
- Source maps for debugging

## Browser Support

- Chrome/Edge (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)

## Troubleshooting

### Backend Connection Issues

```bash
# Check backend is running
curl http://localhost:8000/health

# Verify CORS configuration in backend
```

### Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Port Already in Use

```bash
# Change port in vite.config.js
server: {
  port: 3001  // Use different port
}
```

## Contributing

1. Follow React best practices
2. Use functional components with hooks
3. Maintain consistent code style
4. Add comments for complex logic
5. Test across different browsers

## License

Part of CS685 Adversarial ML Project - Educational purposes

## Links

- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Main Project: ../README.md
