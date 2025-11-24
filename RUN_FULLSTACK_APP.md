# 🌐 Full-Stack Application Guide

**Run your complete Adversarial ML web application with React frontend + FastAPI backend!**

---

## 🎨 What You Have

You have a **complete professional web application** with:

### Frontend (React + Vite + Tailwind CSS)
- ✅ Modern React 18 with hooks
- ✅ Beautiful responsive UI
- ✅ Interactive adversarial attack demo
- ✅ Results dashboard with charts
- ✅ Educational methodology explainer
- ✅ Image upload and real-time processing
- ✅ Gallery of pre-computed results

### Backend (FastAPI + PyTorch)
- ✅ RESTful API
- ✅ FGSM and PGD attack generation
- ✅ Real-time image classification
- ✅ Model serving with PyTorch
- ✅ Automatic API documentation

---

## 🚀 Quick Start (2 Terminals)

### Terminal 1: Start Backend

```powershell
# Navigate to project
cd C:\Users\tanay\OneDrive\Desktop\fds\fds-project

# Activate Python venv
.\venv\Scripts\Activate.ps1

# Start backend API
uvicorn app.backend:app --reload --port 8000
```

**Backend will run on:** http://localhost:8000

### Terminal 2: Start Frontend

```powershell
# Navigate to frontend
cd C:\Users\tanay\OneDrive\Desktop\fds\fds-project\frontend

# Start frontend dev server
npm run dev
```

**Frontend will run on:** http://localhost:5173

---

## 🌟 Access Your Application

Once both are running, open your browser:

### Main Application
**http://localhost:5173**

### Available Pages:

1. **Home** (`/`)
   - Landing page
   - Project overview
   - Features showcase
   - Call-to-action

2. **Interactive Demo** (`/demo`)
   - Upload your own images
   - Choose attack type (FGSM/PGD)
   - Adjust parameters (epsilon, iterations)
   - See real-time results
   - Download adversarial examples

3. **Results Gallery** (`/gallery`)
   - Pre-computed attack statistics
   - Success rate charts
   - Class-wise vulnerability analysis
   - Visual comparisons

4. **About/Methodology** (`/about`)
   - Mathematical explanations
   - Algorithm visualizations
   - Code examples
   - Interactive learning

### Backend API Documentation
**http://localhost:8000/docs** - Interactive Swagger UI

---

## 📊 Frontend Features

### 1. Interactive Attack Demo

**Upload an Image:**
- Drag & drop support
- File validation (JPEG, PNG)
- Image preview
- Or select from CIFAR-10 samples

**Configure Attack:**
```
Attack Type: [FGSM | PGD]
Epsilon (ε): 0.001 ━━━━━●━━━━━ 0.3
```

For PGD:
```
Iterations: 5 ━━━━━●━━━━━ 100
Step Size (α): 0.001 ━━━━━●━━━━━ 0.1
Random Init: [✓]
```

**See Results:**
- Original vs Adversarial image (side-by-side)
- Prediction comparison:
  ```
  Original: "Cat" (95.3% confidence)
  Adversarial: "Dog" (87.1% confidence) ✗
  ```
- Perturbation statistics:
  - L0 norm: 3072 pixels changed
  - L1 norm: 45.23
  - L2 norm: 8.94
  - L∞ norm: 0.030
- Download all images

### 2. Results Dashboard

**Statistical Overview:**
```
┌────────────────────┐ ┌────────────────────┐
│ Model Accuracy     │ │ FGSM Max ASR       │
│      93.45%        │ │      51.82%        │
└────────────────────┘ └────────────────────┘

┌────────────────────┐ ┌────────────────────┐
│ PGD Max ASR        │ │ Model Parameters   │
│      84.76%        │ │   11.17M params    │
└────────────────────┘ └────────────────────┘
```

**Interactive Charts:**
- Attack Success Rate vs Epsilon (line charts)
- Accuracy Degradation (comparison charts)
- Class-wise Vulnerability (bar charts)
- Attack Method Comparison (radar chart)

### 3. Methodology Explainer

**Interactive Tabs:**

📘 **FGSM Tab:**
- Mathematical formula (LaTeX rendered)
- Step-by-step algorithm (animated)
- Python code implementation
- Key insights

📗 **PGD Tab:**
- Mathematical formula (LaTeX rendered)
- Iterative process visualization
- Python code implementation
- Comparison with FGSM

📙 **Comparison Tab:**
- Side-by-side feature comparison
- Detailed comparison table
- Trade-off analysis
- Use case recommendations

### 4. Modern UI/UX

- ✨ Smooth animations
- 📱 Fully responsive (mobile, tablet, desktop)
- 🎨 Professional color scheme
- 🌈 Gradient accents
- 🔄 Real-time updates
- ⚡ Fast Vite dev server
- 🎯 Intuitive navigation
- 💡 Helpful tooltips

---

## 🛠️ Step-by-Step Setup

### Step 1: Verify Requirements

```powershell
# Check Python
python --version
# Should show: Python 3.13.5

# Check Node.js
node --version
# Should show: v16+ or higher

# Check npm
npm --version
# Should show: v10+ or higher
```

### Step 2: Train a Model (if not done)

```powershell
# Quick model (10 epochs) - 20-30 min
python train.py --epochs 10

# OR Full model (200 epochs) - 2-3 hours
python train_baseline.py
```

**Model saved to:** `./models/baseline_best_model.pth`

### Step 3: Start Backend

```powershell
# In project root
.\venv\Scripts\Activate.ps1
uvicorn app.backend:app --reload --port 8000
```

**You should see:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
[OK] Model loaded successfully
```

**Test backend:**
```powershell
# Open in browser
http://localhost:8000/docs
```

### Step 4: Configure Frontend (Optional)

Create `frontend/.env`:
```env
VITE_API_URL=http://localhost:8000
```

### Step 5: Start Frontend

```powershell
# In frontend directory
cd frontend
npm run dev
```

**You should see:**
```
  VITE v5.0.8  ready in 2345 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

### Step 6: Open Application

Open your browser and go to:
**http://localhost:5173**

---

## 🎯 Usage Examples

### Example 1: Generate FGSM Attack

1. Open http://localhost:5173/demo
2. Click "Upload Image" or use sample
3. Select "FGSM" attack type
4. Set epsilon to 0.03
5. Click "Generate Attack"
6. See results in ~1-2 seconds
7. Download images if needed

### Example 2: Compare PGD Iterations

1. Go to demo page
2. Select "PGD" attack type
3. Set epsilon to 0.03
4. Try iterations: 10, 20, 40
5. Compare success rates
6. Notice: More iterations = higher success

### Example 3: Explore Results

1. Go to http://localhost:5173/gallery
2. View attack statistics
3. See charts and visualizations
4. Check class-wise vulnerabilities
5. Understand which classes are most vulnerable

### Example 4: Learn Methodology

1. Go to http://localhost:5173/about
2. Click "FGSM" tab
3. See animated algorithm
4. Read mathematical explanation
5. View code implementation
6. Switch to "PGD" tab
7. Compare in "Comparison" tab

---

## 📁 Project Structure

```
fds-project/
├── app/
│   └── backend.py              ← FastAPI backend
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.jsx        ← Landing page
│   │   │   ├── Demo.jsx        ← Interactive demo
│   │   │   ├── Gallery.jsx     ← Results gallery
│   │   │   └── About.jsx       ← Methodology
│   │   ├── components/
│   │   │   ├── AttackDemo.jsx  ← Main demo component
│   │   │   ├── ResultsDashboard.jsx  ← Charts
│   │   │   ├── MethodologyExplainer.jsx  ← Education
│   │   │   ├── Navbar.jsx      ← Navigation
│   │   │   └── Footer.jsx      ← Footer
│   │   ├── services/
│   │   │   └── api.js          ← API client
│   │   └── App.jsx             ← Main app
│   ├── package.json            ← Dependencies
│   └── vite.config.js          ← Vite config
│
└── models/
    └── baseline_best_model.pth ← Trained model
```

---

## 🔌 API Endpoints (Backend)

The frontend uses these endpoints:

```javascript
GET  /health                 // Health check
GET  /model-info            // Model information
POST /upload                // Upload & classify image
POST /attack                // Generate adversarial attack
GET  /example-results       // Pre-computed results
```

**Example API Call:**
```javascript
// Frontend makes this call
const response = await axios.post('http://localhost:8000/attack', {
  file: imageFile,
  attack_type: 'fgsm',
  epsilon: 0.03
});

// Backend returns
{
  success: true,
  original_prediction: "cat",
  adversarial_prediction: "dog",
  original_confidence: 0.953,
  adversarial_confidence: 0.871,
  attack_success: true,
  original_image: "data:image/jpeg;base64,...",
  adversarial_image: "data:image/jpeg;base64,...",
  perturbation_stats: { ... }
}
```

---

## 🎨 Customization

### Change Frontend Port

Edit `frontend/vite.config.js`:
```javascript
export default defineConfig({
  server: {
    port: 3000  // Change to your preferred port
  }
})
```

### Change Backend Port

```powershell
uvicorn app.backend:app --reload --port 8080
```

Then update `frontend/.env`:
```env
VITE_API_URL=http://localhost:8080
```

### Customize UI Theme

Edit `frontend/tailwind.config.js`:
```javascript
theme: {
  extend: {
    colors: {
      primary: '#your-color',
      secondary: '#your-color'
    }
  }
}
```

---

## 🐛 Troubleshooting

### Backend Issues

**Problem:** "No module named 'torch'"
```powershell
# Solution: Activate venv
.\venv\Scripts\Activate.ps1
```

**Problem:** "Model checkpoint not found"
```powershell
# Solution: Train a model first
python train.py --epochs 10
```

**Problem:** Port 8000 already in use
```powershell
# Solution: Use different port
uvicorn app.backend:app --reload --port 8080
```

### Frontend Issues

**Problem:** "Cannot connect to backend"
```powershell
# Check backend is running
curl http://localhost:8000/health

# Check CORS in backend.py
```

**Problem:** npm install fails
```powershell
# Clear cache and reinstall
cd frontend
rm -r node_modules
rm package-lock.json
npm install
```

**Problem:** Port 5173 already in use
```powershell
# Vite will auto-increment to 5174, 5175, etc.
# Or specify port in vite.config.js
```

### CORS Issues

If frontend can't connect to backend, ensure CORS is configured in `app/backend.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📸 Screenshots & Features

### Home Page
- Hero section with gradient background
- Feature cards with icons
- How it works section
- Call-to-action buttons

### Demo Page
- Split layout (upload | results)
- Parameter sliders with real-time feedback
- Image comparison viewer
- Statistics cards
- Download buttons

### Gallery Page
- KPI cards with metrics
- Interactive Recharts visualizations
- Data table with sorting
- Color-coded success indicators

### About Page
- Tabbed interface
- LaTeX math rendering (KaTeX)
- Syntax-highlighted code
- Animated algorithm flows
- Expandable sections

---

## 🚀 Production Build

### Build Frontend for Production

```powershell
cd frontend
npm run build
```

**Output:** `frontend/dist/` - Ready to deploy!

### Preview Production Build

```powershell
npm run preview
```

### Deploy Options

- **Vercel** - Automatic Vite deployment
- **Netlify** - Drag & drop dist folder
- **GitHub Pages** - Static hosting
- **Your server** - Serve dist folder with nginx/apache

---

## 🎓 Tech Stack Summary

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool & dev server
- **Tailwind CSS** - Utility-first CSS
- **React Router** - Client-side routing
- **Axios** - HTTP client
- **Recharts** - Charts & visualizations
- **KaTeX** - Math rendering
- **Lucide React** - Icons
- **React Syntax Highlighter** - Code display

### Backend
- **FastAPI** - Web framework
- **PyTorch** - ML framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **Pillow** - Image processing

---

## 📊 Performance

### Frontend
- **Dev server:** ~2s startup
- **HMR:** Instant hot reload
- **Build time:** ~10-15s
- **Bundle size:** ~200-300KB (gzipped)

### Backend
- **Startup:** ~3-5s (model loading)
- **FGSM attack:** ~0.5-1s per image
- **PGD attack:** ~1-3s per image (depends on iterations)
- **Image classification:** ~0.2-0.5s

---

## 🎉 You're All Set!

Your complete full-stack adversarial ML application is ready!

**Open:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Next Steps:**
1. Upload an image and generate attacks
2. Explore the results gallery
3. Learn from methodology explainer
4. Customize the UI to your liking

Enjoy your professional ML web application! 🚀

---

**Questions?** Check:
- Frontend README: `frontend/README.md`
- Backend API: `docs/API.md`
- Main README: `README.md`

