# Adversarial ML Project - Demo Script

**Duration**: 3-5 minutes
**Objective**: Demonstrate adversarial attacks on neural networks with interactive visualizations

---

## Pre-Demo Setup Checklist

### Backend Setup
```bash
# Navigate to project directory
cd /path/to/fds-project

# Activate virtual environment
source venv/bin/activate  # or: conda activate cs685

# Start backend server
cd app
python backend.py

# Expected output:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# Model loaded successfully
```

### Frontend Setup
```bash
# In a new terminal
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev

# Expected output:
# VITE v5.0.8  ready in XXX ms
# ➜  Local:   http://localhost:3000/
```

### Verify Everything Works
- Open browser to `http://localhost:3000`
- Check that all pages load without errors
- Verify backend is responding: `curl http://localhost:8000/health`

---

## Demo Script (3-5 minutes)

### Part 1: Introduction (30 seconds)

**What to Say:**
> "This project demonstrates how neural networks can be fooled by adversarial attacks.
> I'll show you how small, carefully crafted perturbations - invisible to the human eye -
> can cause a well-trained model to misclassify images with high confidence."

**What to Show:**
- Home page overview
- Quick scroll to show the interface

---

### Part 2: Interactive Demo (90 seconds)

**Navigate to**: Demo page

**What to Say:**
> "Let's generate an adversarial example in real-time. I'll upload an image of a cat
> and use the PGD attack to create a perturbed version that the model will misclassify."

**Steps to Demonstrate:**

1. **Upload an Image** (10 seconds)
   - Click upload area or drag-and-drop
   - Use: `demo/sample_images/cat_original.png` (prepare this beforehand)
   - Show the image preview appears

2. **Configure Attack Parameters** (15 seconds)
   - Select attack type: **PGD**
   - Set epsilon: **0.03** (8/255)
   - Set iterations: **20**
   - Point out the step size auto-calculates to ε/4

   **What to Say:**
   > "I'm using PGD with epsilon 0.03, which limits perturbations to about 3% of the
   > pixel range. This ensures the changes are imperceptible to humans."

3. **Generate Attack** (30 seconds)
   - Click "Generate Attack" button
   - While loading, explain:
   > "The attack iteratively adjusts pixels in the direction that maximizes the model's
   > loss function. Each iteration brings us closer to fooling the network."

4. **Show Results** (35 seconds)
   - **Point out key metrics:**
     - Original: "cat" (98% confidence) → Adversarial: "dog" (87% confidence)
     - Attack Success: ✓
     - L∞ norm: 0.030 (exactly epsilon)
     - L2 norm: ~0.15

   - **Compare images side-by-side:**
   > "Look at the original and adversarial images - can you see the difference?
   > They look identical to us, but the model is now 87% confident this is a dog!"

   - **Show perturbation (amplified 10x):**
   > "Here's the actual perturbation amplified 10 times. It's structured noise
   > targeting the model's decision boundary."

   - **Download example** (optional):
   > "You can download these images for further analysis."

---

### Part 3: Results Analysis (60 seconds)

**Navigate to**: Gallery page

**What to Say:**
> "Let's look at comprehensive attack results across the CIFAR-10 test set."

**What to Show:**

1. **Stats Cards** (10 seconds)
   - Baseline Accuracy: 91.45%
   - FGSM Attack Success Rate: 79%
   - PGD Attack Success Rate: 91%

   **What to Say:**
   > "Our ResNet18 model achieves 91% accuracy on clean images, but PGD can fool it
   > 91% of the time with imperceptible perturbations."

2. **ASR vs Epsilon Chart** (20 seconds)
   - Show the exponential increase
   - Point out FGSM vs PGD gap

   **What to Say:**
   > "This chart shows attack success rate versus epsilon. Notice PGD is significantly
   > more effective than FGSM, especially at lower epsilon values. At ε=0.03,
   > PGD achieves nearly 90% success rate."

3. **Class Vulnerability** (15 seconds)
   - Show the horizontal bar chart

   **What to Say:**
   > "Some classes are more vulnerable than others. Cats and birds are hardest to
   > defend, while trucks and ships are more robust. This suggests the model has
   > learned more stable features for vehicle classes."

4. **Attack Comparison Table** (15 seconds)
   - Quickly scroll through the comparison table

   **What to Say:**
   > "Here we compare different attack configurations. Notice the trade-off between
   > speed and effectiveness - PGD-40 is most effective but 40x slower than FGSM."

---

### Part 4: Methodology Deep Dive (60 seconds)

**Navigate to**: About page

**What to Say:**
> "Let me quickly explain how these attacks work mathematically."

**What to Show:**

1. **FGSM Tab** (25 seconds)
   - Show the mathematical formula
   - Click "Animate Algorithm Flow"
   - Let animation run through 5 steps

   **What to Say:**
   > "FGSM is a one-step attack. It computes the gradient of the loss with respect
   > to the input, then perturbs in the sign direction. Simple and fast, but less
   > effective than iterative methods."

2. **PGD Tab** (20 seconds)
   - Switch to PGD tab
   - Show the iterative formula
   - Point out the iteration loop diagram

   **What to Say:**
   > "PGD applies multiple small steps, projecting back to the epsilon-ball after
   > each iteration. This iterative refinement makes it much more effective."

3. **Comparison Tab** (15 seconds)
   - Switch to Comparison tab
   - Show side-by-side cards

   **What to Say:**
   > "The key trade-off: FGSM is 10-20x faster but 50% less effective. For security
   > evaluation, we use PGD as the gold standard despite the computational cost."

---

### Part 5: Conclusion (20 seconds)

**What to Say:**
> "This project demonstrates the critical vulnerability of neural networks to
> adversarial attacks. Understanding these attacks is essential for building
> robust AI systems, especially in security-critical applications like autonomous
> vehicles and medical diagnosis."

**Optional**: Return to Home page for clean ending

---

## Key Talking Points to Emphasize

### Technical Depth
- ✓ Gradient-based white-box attacks
- ✓ L∞ norm constraints (bounded perturbations)
- ✓ Iterative optimization (PGD)
- ✓ Projection to feasible set

### Practical Implications
- ✓ Security risks in deployed ML systems
- ✓ Need for adversarial training
- ✓ Defense mechanisms (future work)

### Implementation Quality
- ✓ Interactive web application
- ✓ Real-time attack generation
- ✓ Comprehensive visualizations
- ✓ Educational value

---

## Sample Images to Use

### Recommended Test Images

**High Success Rate (Good for Demo):**
- `cat_original.png` - Cat → Dog (classic adversarial example)
- `bird_original.png` - Bird → Airplane
- `deer_original.png` - Deer → Horse

**Interesting Failures:**
- `truck_robust.png` - Truck remains truck (shows robustness)
- `ship_resistant.png` - Ship hard to attack

**Prepare 2-3 images beforehand** in `demo/sample_images/` directory

---

## Common Issues & Troubleshooting

### Backend Issues

**Problem**: "Model file not found"
```bash
# Solution: Check model checkpoint exists
ls checkpoints/

# If missing, run training or download pre-trained
python train.py
```

**Problem**: "CUDA out of memory"
```python
# Solution: In backend.py, force CPU mode
device = torch.device('cpu')
```

**Problem**: "Port 8000 already in use"
```bash
# Solution: Kill existing process or use different port
lsof -ti:8000 | xargs kill -9
# Or change port in backend.py: uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Frontend Issues

**Problem**: "Module not found" errors
```bash
# Solution: Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Problem**: "API connection failed"
```bash
# Solution: Check backend is running and CORS is enabled
curl http://localhost:8000/health

# Verify proxy in vite.config.js
```

**Problem**: Katex/charts not rendering
```bash
# Solution: Clear cache and rebuild
npm run build
npm run dev
```

### Demo Issues

**Problem**: Attack generation takes too long
- Use smaller images (CIFAR-10 size: 32×32)
- Reduce PGD iterations to 10 instead of 40
- Pre-generate examples and show those

**Problem**: Results not visually interesting
- Choose images with high attack success rate
- Use epsilon = 0.03 for clear but imperceptible perturbations
- Show perturbation with 10x amplification

**Problem**: Audience can't see small images
- Zoom browser to 150-200%
- Use fullscreen mode (F11)
- Prepare screenshots for slides as backup

---

## Timing Breakdown

| Section | Time | Details |
|---------|------|---------|
| Introduction | 0:00 - 0:30 | Project overview |
| Interactive Demo | 0:30 - 2:00 | Live attack generation |
| Results Analysis | 2:00 - 3:00 | Charts and metrics |
| Methodology | 3:00 - 4:00 | Algorithm explanation |
| Conclusion | 4:00 - 4:20 | Wrap-up |
| **Total** | **4:20** | Leaves buffer for questions |

---

## Post-Demo Q&A Preparation

### Expected Questions

**Q: Can these attacks work on real-world systems?**
> Yes, adversarial examples can transfer between models. Physical adversarial
> patches have been used to fool object detection systems in the real world.

**Q: How do we defend against these attacks?**
> Main defense: adversarial training - training on adversarial examples. Other
> methods include input transformation, certified defenses, and ensemble methods.

**Q: Why do adversarial examples exist?**
> Neural networks learn non-robust features and operate in high-dimensional space
> where small perturbations can cause large changes in output. The decision
> boundaries are surprisingly close to natural data points.

**Q: How long did this project take?**
> [Adjust based on your timeline] - Training models, implementing attacks, building
> the web interface, and creating visualizations took approximately [X weeks/months].

**Q: What challenges did you face?**
> Key challenges: efficient batch attack generation, creating intuitive visualizations
> for complex mathematical concepts, ensuring reproducibility, and building a
> responsive web interface.

---

## Video Recording Tips

### Camera & Audio
- Use good lighting (face the window or use a lamp)
- Test microphone beforehand (clear audio is critical)
- Use screen recording software (OBS, QuickTime, or Zoom)
- Record at 1080p minimum

### Screen Setup
- Close unnecessary tabs and applications
- Disable notifications
- Set browser zoom to 100-125% for visibility
- Use fullscreen mode for demo pages
- Have a clean desktop

### Presentation Tips
- Speak clearly and at moderate pace
- Use pointer/cursor to highlight important elements
- Pause briefly between sections
- If you make a mistake, just continue (can edit later)
- Practice 2-3 times before final recording

### Editing
- Cut out long loading times (or speed up 2x)
- Add text overlays for key concepts
- Include captions if possible
- Add intro/outro slides with project info

---

## Backup Plan

### If Live Demo Fails
1. **Have screenshots ready**: Prepare 8-10 screenshots showing key results
2. **Pre-record video**: Have a backup recording of the demo
3. **Static presentation**: Create slides as fallback

### Quick Demo Recovery
- Keep backend running in background
- Have browser tabs pre-loaded
- Keep sample images ready to upload
- Have backup images if one doesn't work well

---

## Next Steps After Demo

### Suggested Improvements to Mention
1. Real-time iteration visualization for PGD
2. Multi-model comparison (ResNet vs VGG vs LeNet)
3. Batch processing for multiple images
4. Defense mechanisms (adversarial training)
5. Black-box attacks (transferability)
6. Physical adversarial patches

### GitHub Repository
- Make repository public before demo
- Include comprehensive README
- Add requirements.txt and setup instructions
- Include license and citation information

---

## Success Criteria

✓ Demo runs smoothly without major technical issues
✓ Audience understands the concept of adversarial attacks
✓ Visual results are compelling and clear
✓ Time stays within 5 minutes
✓ Questions are handled confidently

Good luck with your demo! 🎯
