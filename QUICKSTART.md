# 🚀 Quick Start Guide - First Run

This guide ensures everything works perfectly on the first run.

## Automated Setup (Recommended)

Run the automated setup script:

```bash
cd /home/user/fds-project
./setup_and_verify.sh
```

This script will:
- ✓ Check Python and Node.js installations
- ✓ Install all dependencies (Python + npm)
- ✓ Verify all imports work
- ✓ Check project structure
- ✓ Create necessary directories
- ✓ Display next steps

**Expected time: 3-5 minutes**

---

## Manual Setup (Alternative)

If you prefer manual setup:

### 1. Install Python Dependencies

```bash
cd /home/user/fds-project

# Recommended: Use virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision fastapi uvicorn python-multipart pillow numpy
```

### 2. Install Frontend Dependencies

```bash
cd /home/user/fds-project/frontend

# Install all npm packages (including katex, recharts, react-syntax-highlighter)
npm install
```

**Expected time: 3-5 minutes**

---

## Running the Application

### Terminal 1: Start Backend

```bash
cd /home/user/fds-project/app
python backend.py
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Port:** 8000
**URL:** http://localhost:8000

### Terminal 2: Start Frontend

```bash
cd /home/user/fds-project/frontend
npm run dev
```

**Expected output:**
```
VITE v5.0.8  ready in 500 ms
➜  Local:   http://localhost:3000/
```

**Port:** 3000
**URL:** http://localhost:3000

---

## First-Time Verification

### 1. Quick Health Check

```bash
# Test backend is running
curl http://localhost:8000/health

# Expected: {"status":"healthy","model_loaded":true}
```

### 2. Open Browser

Navigate to: **http://localhost:3000**

You should see:
- ✓ Home page with project overview
- ✓ Navbar with Demo, Gallery, About links
- ✓ No console errors (press F12 to check)

### 3. Test Each Feature

**a) Single Attack:**
1. Click **Demo** → **Single Attack** tab
2. Upload any image (PNG/JPG)
3. Click **Generate Attack**
4. Verify you see before/after images

**b) Real-Time Visualization:** ⭐ (Most impressive!)
1. Click **Real-Time Visualization** tab
2. Upload an image
3. Click **Generate Attack**
4. Click **Play** button
5. Watch attack animate iteration-by-iteration!

**c) Batch Attack:**
1. Click **Batch Attack** tab
2. Upload 3-5 images
3. Click **Generate Batch Attack**
4. Verify summary statistics appear
5. Test CSV/JSON export

**d) Results Gallery:**
1. Click **Gallery** in navbar
2. Verify charts render (may show dummy data initially)

**e) Methodology:**
1. Click **About** in navbar
2. Verify math formulas render
3. Click **Animate Algorithm Flow** button

---

## Running Tests

### Quick Test (2 minutes)

```bash
cd /home/user/fds-project
python demo/test_all.py --quick
```

### Full Test Suite (5-10 minutes)

```bash
python demo/test_all.py
```

### Expected Output

```
======================================================================
                         Model Loading Tests
======================================================================

✓ PASS | Model initialization
✓ PASS | Checkpoint loading
...

Test Summary
Total Tests: 20
Passed: 20
Failed: 0
Pass Rate: 100.0%

🎉 All tests passed!
```

---

## Troubleshooting Common Issues

### Issue: "Module not found: fastapi"

**Solution:**
```bash
pip install fastapi uvicorn python-multipart pillow torch torchvision numpy
```

### Issue: "Cannot find module 'katex'"

**Solution:**
```bash
cd frontend
npm install
```

### Issue: "Port 8000 already in use"

**Solution:**
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Or use different port
cd app
python backend.py --port 8001
```

Then update frontend API URL:
```bash
# Edit frontend/.env
echo "VITE_API_URL=http://localhost:8001" > frontend/.env
```

### Issue: "Port 3000 already in use"

**Solution:**
```bash
# Kill existing process
lsof -ti:3000 | xargs kill -9

# Or Vite will prompt you to use a different port
```

### Issue: Model not found

**Don't worry!** The backend works with dummy data if no model is found.

To train a model:
```bash
python train.py
```

This creates `models/baseline_best_model.pth`

### Issue: Math formulas not rendering

**Solution:**
```bash
cd frontend
npm install katex react-katex
npm run dev
```

### Issue: Charts not showing

**Solution:**
```bash
cd frontend
npm install recharts
npm run dev
```

### Issue: Code syntax highlighting not working

**Solution:**
```bash
cd frontend
npm install react-syntax-highlighter
npm run dev
```

---

## File Checklist

Verify these files exist:

**Backend:**
- ✓ `app/backend.py`
- ✓ `app/__init__.py`

**Frontend Components:**
- ✓ `frontend/src/components/AttackDemo.jsx`
- ✓ `frontend/src/components/BatchAttack.jsx` ⭐ NEW
- ✓ `frontend/src/components/RealTimeAttack.jsx` ⭐ NEW
- ✓ `frontend/src/components/ResultsDashboard.jsx`
- ✓ `frontend/src/components/MethodologyExplainer.jsx`

**Frontend Pages:**
- ✓ `frontend/src/pages/Demo.jsx` (updated with 3 tabs)
- ✓ `frontend/src/pages/Gallery.jsx`
- ✓ `frontend/src/pages/About.jsx`

**Services:**
- ✓ `frontend/src/services/api.js` (updated with new endpoints)

**Demo:**
- ✓ `demo/demo_script.md`
- ✓ `demo/test_all.py`

---

## Dependencies Installed

**Python packages:**
- torch, torchvision (PyTorch)
- fastapi (Web framework)
- uvicorn (ASGI server)
- python-multipart (File uploads)
- pillow (Image processing)
- numpy (Numerical operations)

**npm packages:**
- react, react-dom, react-router-dom (React framework)
- axios (HTTP client)
- lucide-react (Icons)
- recharts (Charts)
- katex, react-katex (Math formulas) ⭐ NEW
- react-syntax-highlighter (Code highlighting) ⭐ NEW
- vite (Build tool)
- tailwindcss (CSS framework)

---

## API Endpoints Available

Test these endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info

# Single attack
curl -X POST -F "file=@image.png" -F "attack_type=fgsm" http://localhost:8000/attack

# Batch attack (NEW)
curl -X POST -F "files=@img1.png" -F "files=@img2.png" http://localhost:8000/batch-attack

# Attack iterations (NEW - for real-time viz)
curl -X POST -F "file=@image.png" -F "pgd_iterations=20" http://localhost:8000/attack-iterations

# Example results
curl http://localhost:8000/example-results
```

---

## Performance Tips

### Speed up demos:

1. **Reduce PGD iterations:**
   - Default: 20 iterations (~30s)
   - Fast demo: 10 iterations (~15s)
   - Edit in frontend components or pass as parameter

2. **Use FGSM for quick tests:**
   - FGSM: ~2 seconds
   - PGD: ~30 seconds

3. **Pre-load images:**
   - Have 3-5 test images ready
   - CIFAR-10 size (32×32) works best

---

## Demo Checklist

Before recording video:

- [ ] Run `./setup_and_verify.sh` ✓
- [ ] Run `python demo/test_all.py --quick` ✓
- [ ] Backend running on port 8000 ✓
- [ ] Frontend running on port 3000 ✓
- [ ] Open `http://localhost:3000` in browser ✓
- [ ] Test single attack works ✓
- [ ] Test real-time visualization works ✓
- [ ] Test batch attack works ✓
- [ ] Close unnecessary apps ✓
- [ ] Disable notifications ✓
- [ ] Test microphone ✓
- [ ] Have sample images ready ✓
- [ ] Review `demo/demo_script.md` ✓

---

## Success Indicators

✅ **Everything is working if:**

1. Backend starts without errors
2. Frontend starts without errors
3. No console errors in browser (F12)
4. You can upload an image in Demo page
5. Attack generates and shows results
6. Real-time visualization plays/pauses
7. Batch attack processes multiple images
8. Charts render in Gallery page
9. Math formulas show in About page
10. All test pass: `python demo/test_all.py`

---

## Getting Help

**Check these first:**
1. Backend logs in terminal 1
2. Frontend logs in terminal 2
3. Browser console (F12 → Console)
4. Network tab (F12 → Network)

**Common fixes:**
- Restart backend and frontend
- Clear browser cache (Ctrl+Shift+R)
- Re-run `npm install` in frontend/
- Re-run `pip install` for Python packages

---

## 🎉 You're Ready!

Once everything is running:

1. Open `demo/demo_script.md` for exact demo talking points
2. Start with **Real-Time Visualization** - most impressive!
3. Show **Batch Attack** for scalability
4. Explain **Methodology** with animated algorithms

**The real-time visualization is perfect for video demos - it dramatically shows how PGD iteratively fools the neural network!**

Good luck! 🚀
