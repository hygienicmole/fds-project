#!/bin/bash
# Complete Setup and Verification Script for Adversarial ML Project
# This ensures everything works perfectly on the first run

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Adversarial ML Project Setup${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Get project root
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_ROOT"

# Step 1: Check Python
echo -e "${YELLOW}[1/8] Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Python found: $PYTHON_VERSION${NC}"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    echo -e "${GREEN}✓ Python found: $PYTHON_VERSION${NC}"
    PYTHON_CMD="python"
else
    echo -e "${RED}✗ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Step 2: Check Node.js and npm
echo -e "\n${YELLOW}[2/8] Checking Node.js and npm...${NC}"
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓ Node.js found: $NODE_VERSION${NC}"
else
    echo -e "${RED}✗ Node.js not found. Please install Node.js 16+${NC}"
    exit 1
fi

if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓ npm found: v$NPM_VERSION${NC}"
else
    echo -e "${RED}✗ npm not found. Please install npm${NC}"
    exit 1
fi

# Step 3: Install Python dependencies
echo -e "\n${YELLOW}[3/8] Installing Python dependencies...${NC}"
echo "This may take a few minutes..."

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠ Not in a virtual environment. Installing globally.${NC}"
    echo -e "${YELLOW}  Consider using: python3 -m venv venv && source venv/bin/activate${NC}"
fi

# Install required packages
$PYTHON_CMD -m pip install --upgrade pip --quiet
$PYTHON_CMD -m pip install \
    torch torchvision \
    fastapi uvicorn python-multipart \
    pillow numpy \
    --quiet

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install Python dependencies${NC}"
    exit 1
fi

# Step 4: Verify Python imports
echo -e "\n${YELLOW}[4/8] Verifying Python imports...${NC}"
$PYTHON_CMD -c "
import torch
import torchvision
import fastapi
import uvicorn
import PIL
import numpy
print('All imports successful')
" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All Python imports working${NC}"
else
    echo -e "${RED}✗ Python import verification failed${NC}"
    exit 1
fi

# Step 5: Install frontend dependencies
echo -e "\n${YELLOW}[5/8] Installing frontend dependencies...${NC}"
echo "This may take a few minutes..."

cd frontend
npm install --quiet --legacy-peer-deps 2>&1 | tail -5

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install frontend dependencies${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"

# Step 6: Verify project structure
echo -e "\n${YELLOW}[6/8] Verifying project structure...${NC}"

REQUIRED_DIRS=(
    "app"
    "frontend/src/components"
    "frontend/src/pages"
    "frontend/src/services"
    "demo"
    "models"
    "attacks"
)

REQUIRED_FILES=(
    "app/backend.py"
    "frontend/src/App.jsx"
    "frontend/src/components/AttackDemo.jsx"
    "frontend/src/components/BatchAttack.jsx"
    "frontend/src/components/RealTimeAttack.jsx"
    "frontend/src/components/ResultsDashboard.jsx"
    "frontend/src/components/MethodologyExplainer.jsx"
    "frontend/src/pages/Demo.jsx"
    "frontend/src/pages/Gallery.jsx"
    "frontend/src/pages/About.jsx"
    "frontend/src/services/api.js"
    "demo/demo_script.md"
    "demo/test_all.py"
)

ALL_GOOD=true

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓ $dir${NC}"
    else
        echo -e "${RED}✗ Missing directory: $dir${NC}"
        ALL_GOOD=false
    fi
done

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ Missing file: $file${NC}"
        ALL_GOOD=false
    fi
done

if [ "$ALL_GOOD" = false ]; then
    echo -e "${RED}✗ Project structure incomplete${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Project structure verified${NC}"
fi

# Step 7: Check for model file
echo -e "\n${YELLOW}[7/8] Checking for trained model...${NC}"
if [ -f "models/baseline_best_model.pth" ]; then
    MODEL_SIZE=$(du -h models/baseline_best_model.pth | cut -f1)
    echo -e "${GREEN}✓ Model found: models/baseline_best_model.pth ($MODEL_SIZE)${NC}"
elif [ -f "checkpoints/best_model.pth" ]; then
    MODEL_SIZE=$(du -h checkpoints/best_model.pth | cut -f1)
    echo -e "${GREEN}✓ Model found: checkpoints/best_model.pth ($MODEL_SIZE)${NC}"
else
    echo -e "${YELLOW}⚠ No trained model found${NC}"
    echo -e "${YELLOW}  Backend will work but you need to train a model first:${NC}"
    echo -e "${YELLOW}  python train.py${NC}"
    echo -e "${YELLOW}  Or the backend will use dummy data${NC}"
fi

# Step 8: Create necessary directories
echo -e "\n${YELLOW}[8/8] Creating necessary directories...${NC}"
mkdir -p results/attack_evaluation
mkdir -p checkpoints
mkdir -p data
echo -e "${GREEN}✓ Directories created${NC}"

# Final summary
echo -e "\n${BLUE}======================================${NC}"
echo -e "${BLUE}Setup Complete!${NC}"
echo -e "${BLUE}======================================${NC}\n"

echo -e "${GREEN}✓ All dependencies installed${NC}"
echo -e "${GREEN}✓ Project structure verified${NC}"
echo -e "${GREEN}✓ Ready to run!${NC}\n"

# Display next steps
echo -e "${YELLOW}Next Steps:${NC}\n"
echo -e "1. ${BLUE}Run automated tests:${NC}"
echo -e "   $PYTHON_CMD demo/test_all.py --quick\n"

echo -e "2. ${BLUE}Start backend (in one terminal):${NC}"
echo -e "   cd app"
echo -e "   $PYTHON_CMD backend.py\n"

echo -e "3. ${BLUE}Start frontend (in another terminal):${NC}"
echo -e "   cd frontend"
echo -e "   npm run dev\n"

echo -e "4. ${BLUE}Open in browser:${NC}"
echo -e "   http://localhost:3000\n"

echo -e "${YELLOW}📖 For detailed demo instructions, see:${NC}"
echo -e "   demo/demo_script.md\n"

echo -e "${GREEN}🎉 Everything is ready! Good luck with your demo!${NC}\n"
