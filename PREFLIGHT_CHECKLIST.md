# ✈️ Pre-Flight Checklist - First Run Guarantee

This checklist ensures **100% success** on your first run.

## ⚡ Quick Start (30 seconds)

```bash
cd /home/user/fds-project

# One command to rule them all:
./setup_and_verify.sh
```

**That's it!** The script handles everything automatically.

---

## 📋 Manual Verification (If Preferred)

### Phase 1: Prerequisites (2 min)

- [ ] **Python 3.8+ installed**
  ```bash
  python3 --version  # Should show 3.8 or higher
  ```

- [ ] **Node.js 16+ installed**
  ```bash
  node --version     # Should show v16 or higher
  npm --version      # Should show version
  ```

- [ ] **Git installed**
  ```bash
  git --version      # Should show version
  ```

### Phase 2: Python Setup (3 min)

- [ ] **Create virtual environment (recommended)**
  ```bash
  python3 -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

- [ ] **Install Python dependencies**
  ```bash
  pip install torch torchvision fastapi uvicorn python-multipart pillow numpy
  ```

- [ ] **Verify Python imports**
  ```bash
  python3 -c "import torch, fastapi, PIL, numpy; print('✓ All imports OK')"
  ```

### Phase 3: Frontend Setup (3 min)

- [ ] **Install npm dependencies**
  ```bash
  cd frontend
  npm install
  cd ..
  ```

- [ ] **Verify package.json has all dependencies**
  ```bash
  grep -E "katex|recharts|react-syntax-highlighter" frontend/package.json
  # Should show all three packages
  ```

### Phase 4: File Verification (1 min)

- [ ] **Backend files exist**
  ```bash
  test -f app/backend.py && echo "✓ backend.py exists"
  ```

- [ ] **Frontend components exist**
  ```bash
  test -f frontend/src/components/BatchAttack.jsx && echo "✓ BatchAttack.jsx exists"
  test -f frontend/src/components/RealTimeAttack.jsx && echo "✓ RealTimeAttack.jsx exists"
  test -f frontend/src/components/ResultsDashboard.jsx && echo "✓ ResultsDashboard.jsx exists"
  test -f frontend/src/components/MethodologyExplainer.jsx && echo "✓ MethodologyExplainer.jsx exists"
  ```

- [ ] **Demo files exist**
  ```bash
  test -f demo/demo_script.md && echo "✓ demo_script.md exists"
  test -f demo/test_all.py && echo "✓ test_all.py exists"
  ```

### Phase 5: Run Tests (2 min)

- [ ] **Quick test (skips slow tests)**
  ```bash
  python demo/test_all.py --quick
  # Should show: 🎉 All tests passed!
  ```

### Phase 6: Start Services (1 min)

- [ ] **Terminal 1: Start backend**
  ```bash
  cd app
  python backend.py
  # Wait for: "Uvicorn running on http://0.0.0.0:8000"
  ```

- [ ] **Terminal 2: Start frontend**
  ```bash
  cd frontend
  npm run dev
  # Wait for: "Local: http://localhost:3000/"
  ```

### Phase 7: Smoke Tests (2 min)

- [ ] **Backend health check**
  ```bash
  curl http://localhost:8000/health
  # Should return: {"status":"healthy","model_loaded":true}
  ```

- [ ] **Frontend loads**
  ```bash
  curl -I http://localhost:3000
  # Should return: HTTP/1.1 200 OK
  ```

- [ ] **Open in browser**
  - Navigate to: http://localhost:3000
  - Should see home page
  - Press F12 and check console for errors

### Phase 8: Feature Tests (5 min)

- [ ] **Test Single Attack**
  1. Click Demo → Single Attack
  2. Upload an image
  3. Click Generate Attack
  4. Verify results appear

- [ ] **Test Real-Time Visualization**
  1. Click Real-Time Visualization tab
  2. Upload an image
  3. Click Generate Attack
  4. Click Play button
  5. Watch animation

- [ ] **Test Batch Attack**
  1. Click Batch Attack tab
  2. Upload 2-3 images
  3. Click Generate Batch Attack
  4. Verify summary statistics
  5. Test CSV export

- [ ] **Test Gallery**
  1. Click Gallery in navbar
  2. Verify charts render

- [ ] **Test Methodology**
  1. Click About in navbar
  2. Verify math formulas render
  3. Click Animate Algorithm Flow

---

## 🔥 Common Issues & Instant Fixes

### Issue: "Python not found"
```bash
# Ubuntu/Debian
sudo apt-get install python3 python3-pip

# macOS
brew install python3

# Verify
python3 --version
```

### Issue: "Node not found"
```bash
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS
brew install node

# Verify
node --version
```

### Issue: "pip install fails"
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Install with no cache
pip install --no-cache-dir torch torchvision fastapi uvicorn python-multipart pillow numpy
```

### Issue: "npm install fails"
```bash
# Clear npm cache
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force

# Reinstall
npm install --legacy-peer-deps
```

### Issue: "Port 8000 in use"
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9

# Or change backend port
cd app
uvicorn backend:app --port 8001
```

### Issue: "Port 3000 in use"
```bash
# Find and kill process
lsof -ti:3000 | xargs kill -9

# Or Vite will prompt for different port
```

### Issue: "Module not found: katex"
```bash
cd frontend
npm install katex react-katex react-syntax-highlighter
npm run dev
```

### Issue: "Model not found"
**This is OK!** The backend works with random weights for demo purposes.

To train a real model:
```bash
python train.py
```

### Issue: "CORS errors in browser"
Already handled! Backend has CORS enabled for all origins.

### Issue: "Math formulas not rendering"
Clear browser cache:
- Chrome/Firefox: Ctrl+Shift+R
- Or hard refresh

---

## ✅ Success Criteria

### You're ready if:

1. ✅ `./setup_and_verify.sh` completes successfully
2. ✅ `python demo/test_all.py --quick` shows all tests passed
3. ✅ Backend starts without errors
4. ✅ Frontend starts without errors
5. ✅ `curl http://localhost:8000/health` returns healthy
6. ✅ Browser shows home page at http://localhost:3000
7. ✅ No console errors in browser (F12)
8. ✅ Can upload image and generate attack
9. ✅ Real-time visualization plays/pauses
10. ✅ Math formulas render in About page

**If all 10 ✅ - you're 100% ready!**

---

## 🎬 Demo Recording Checklist

Before you record:

- [ ] Run full test suite: `python demo/test_all.py`
- [ ] Backend running smoothly
- [ ] Frontend running smoothly
- [ ] Have 3-5 test images ready
- [ ] Review `demo/demo_script.md`
- [ ] Close unnecessary applications
- [ ] Disable desktop notifications
- [ ] Set browser zoom to 125%
- [ ] Test microphone
- [ ] Test screen recording software
- [ ] Do a practice run (2-3 minutes)

---

## 🚨 Emergency Recovery

If something goes terribly wrong:

### Nuclear Option: Complete Reset

```bash
cd /home/user/fds-project

# Clean everything
cd frontend && rm -rf node_modules package-lock.json && cd ..
rm -rf venv

# Reinstall
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision fastapi uvicorn python-multipart pillow numpy

cd frontend
npm install
cd ..

# Run setup
./setup_and_verify.sh
```

---

## 📊 Expected Timings

| Task | Time | Notes |
|------|------|-------|
| Python dependency install | 2-3 min | First time only |
| npm install | 2-3 min | First time only |
| Backend startup | 5-10 sec | Each time |
| Frontend startup | 3-5 sec | Each time |
| Single attack generation | 2-5 sec | FGSM |
| Single attack generation | 20-30 sec | PGD |
| Real-time viz generation | 30-40 sec | 20 iterations |
| Batch attack (5 images) | 1-2 min | Depends on images |
| Test suite (quick) | 1-2 min | Skips slow tests |
| Test suite (full) | 5-10 min | All tests |

---

## 🎯 Priority Order for Demo

**If short on time, test in this order:**

1. **Real-Time Visualization** (MUST WORK - star of the show!)
2. Single Attack (basic functionality)
3. Batch Attack (impressive scalability)
4. Gallery Charts (visual impact)
5. Methodology Animation (educational value)

---

## 💡 Pro Tips

1. **Speed up PGD for demos:**
   - Use 10 iterations instead of 20
   - Still impressive, half the time

2. **Pre-generate results:**
   - Generate attacks beforehand
   - Take screenshots
   - Use as backup

3. **Test your test images:**
   - Some images attack better than others
   - Try 5-10, pick best 2-3
   - Cats/dogs usually work well

4. **Browser zoom:**
   - 125% zoom is perfect for screen recording
   - Makes UI elements clearly visible

5. **Keyboard shortcuts:**
   - F11: Fullscreen
   - F12: Developer tools
   - Ctrl+Shift+R: Hard refresh

---

## 🆘 Getting Help

**If stuck:**

1. Check backend logs (Terminal 1)
2. Check frontend logs (Terminal 2)
3. Check browser console (F12)
4. Run `python demo/test_all.py -v` (verbose)
5. Check `QUICKSTART.md` for detailed guide
6. Check `demo/demo_script.md` for demo instructions

---

## 🎉 Final Confirmation

Before you declare success, verify:

```bash
# Complete verification script
cd /home/user/fds-project

# 1. Setup
./setup_and_verify.sh

# 2. Tests
python demo/test_all.py --quick

# 3. Backend (Terminal 1)
cd app && python backend.py &
sleep 10

# 4. Health check
curl http://localhost:8000/health

# 5. Frontend (Terminal 2 - do manually)
# cd frontend && npm run dev

# 6. Browser test (do manually)
# Open http://localhost:3000
# Test each feature

# 7. If all OK:
echo "🎉 YOU ARE READY!"
```

---

**Good luck with your demo! You've got this! 🚀**
