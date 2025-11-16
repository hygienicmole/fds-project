# Report Generation Guide

This guide explains how to use the comprehensive report generator to aggregate all experimental results and create publication-ready outputs.

## Overview

The `ReportGenerator` class automatically aggregates results from all experiments and generates:
1. **LaTeX-formatted tables** for academic reports
2. **Markdown summary** with all metrics and findings
3. **Figure listings** with paths and captions
4. **Key observations** and recommendations

## Quick Start

```bash
# Generate all reports (uses default paths)
python utils/report_generator.py

# Custom paths
python utils/report_generator.py \
    --results-dir ./results \
    --output-dir ./my_reports
```

Or in Python:

```python
from utils.report_generator import ReportGenerator

generator = ReportGenerator()
generator.generate_all_reports()
```

## Input Data Sources

The report generator automatically finds and loads:

### 1. Attack Evaluation Results
- **File**: `results/attack_evaluation/attack_results.json`
- **Contains**: Clean accuracy, FGSM results, PGD results with multiple configurations

### 2. Hyperparameter Analysis Results
- **File**: `results/hyperparameter_analysis/results.json`
- **Contains**: Epsilon sweep data, iteration analysis, statistical measures

### 3. Baseline Training Metrics
- **File**: `results/baseline_training/baseline_metrics.json`
- **Contains**: Training history, best model info, timing data

### 4. Visualizations
- **Directory**: `results/` (all subdirectories)
- **Contains**: All `.png` figure files

## Generated Outputs

### 1. LaTeX Tables (`reports/latex_tables.tex`)

**Contents**:
Five professionally formatted LaTeX tables ready for inclusion in reports:

#### Table 1: Attack Evaluation Results
```latex
\begin{table}[h]
\centering
\caption{Model Robustness Against Adversarial Attacks}
\label{tab:attack_eval}
\begin{tabular}{lccccc}
\toprule
Attack & $\epsilon$ & Clean Acc (\%) & Adv Acc (\%) & ASR (\%) & Acc Drop (\%) \\
\midrule
None & 0.000 & 91.45 & 91.45 & 0.00 & 0.00 \\
FGSM & 0.010 & 91.45 & 78.23 & 13.22 & 13.22 \\
...
\bottomrule
\end{tabular}
\end{table}
```

Includes all tested configurations with proper formatting.

#### Table 2: FGSM Epsilon Analysis
Detailed breakdown of FGSM attack effectiveness across epsilon values, showing:
- Epsilon values (normalized and 0-255 scale)
- Adversarial accuracy
- Attack success rate
- L∞ perturbation norms

#### Table 3: PGD Iteration Effect
Analysis of how PGD iterations affect attack success:
- Number of iterations
- Adversarial accuracy
- Attack success rate
- Improvement over previous iteration count
- Relative time cost

#### Table 4: Baseline Training Results
Summary of model training:
- Best validation accuracy and epoch
- Training/validation accuracy at best epoch
- Final accuracies
- Training time
- Model parameters

#### Table 5: Hyperparameter Analysis Summary
Key findings from hyperparameter sweep:
- Representative epsilon values
- FGSM vs PGD-20 comparison
- Demonstrates attack strength difference

**Usage in LaTeX**:
```latex
% In your report.tex file
\input{reports/latex_tables.tex}
```

**Required LaTeX packages**:
```latex
\usepackage{booktabs}  % For professional tables
\usepackage{multirow}  % For complex tables (if needed)
```

---

### 2. Markdown Summary (`reports/EXPERIMENTAL_SUMMARY.md`)

**Contents**:
Comprehensive summary with 6 main sections:

#### Section 1: Executive Summary
- High-level overview of all results
- Baseline performance
- Attack results at standard epsilon (0.03)
- Key finding statement

#### Section 2: Baseline Model Performance
- Architecture details
- Training configuration
- Best and final accuracies
- Training time and parameters

#### Section 3: Attack Evaluation Results
- Clean accuracy
- **FGSM table**: All epsilon values with metrics
- **PGD table**: All configurations with metrics

Example FGSM table:
```markdown
| Epsilon | ε (0-255) | Clean Acc | Adv Acc | ASR | Acc Drop |
|---------|-----------|-----------|---------|-----|----------|
| 0.0100  | 2.55      | 91.45%    | 78.23%  | 13.22% | 13.22% |
| 0.0300  | 7.65      | 91.45%    | 52.34%  | 39.11% | 39.11% |
...
```

#### Section 4: Hyperparameter Analysis
- Epsilon range tested
- **FGSM vs PGD-20 comparison table**
- PGD iteration effect analysis
- Statistical insights

#### Section 5: Key Findings and Observations
**Critical Observations**:
1. Severe vulnerability to adversarial attacks
2. PGD significantly stronger than FGSM
3. Epsilon-robustness relationship
4. Model has no adversarial robustness
5. Attack success varies by class

**Statistical Significance**: All results with p-values and reproducibility notes

#### Section 6: Generated Figures
Complete listing of all figures organized by category:

- **Training Visualizations**: Loss/accuracy curves
- **Attack Evaluation**: Comparison plots, visual grids
- **Hyperparameter Analysis**: Epsilon analysis, iteration effects
- **Comprehensive Visualizations**: Confusion matrices, class-wise analysis, confidence distributions

Each figure entry includes:
- File name
- Descriptive caption
- Relative path

Example:
```markdown
### Hyperparameter Analysis Visualizations

- **epsilon_analysis.png**: 4-panel epsilon effect analysis (accuracy, ASR, drop, log scale)
  - Path: `results/hyperparameter_analysis/epsilon_analysis.png`

- **iteration_analysis.png**: PGD iteration effect on attack success
  - Path: `results/hyperparameter_analysis/iteration_analysis.png`
```

#### Section 7: Recommendations
**For Future Work**:
- Adversarial training implementation
- Certified defenses
- Additional attacks
- Other datasets
- Architecture comparisons

**For Deployment**:
- Critical warnings about standard models
- Security best practices
- Monitoring recommendations

#### Appendix: Detailed Metrics
- Links to raw JSON files
- Statistical measures included
- Experimental configuration details

---

## Example Workflow

### Step 1: Run All Experiments

```bash
# Train baseline model
python train_baseline.py

# Evaluate attacks
python attacks/evaluate_attacks.py

# Analyze hyperparameters
python experiments/hyperparameter_analysis.py

# Generate visualizations
python generate_visualizations.py --attack fgsm
python generate_visualizations.py --attack pgd
```

### Step 2: Generate Reports

```bash
# Generate all reports
python utils/report_generator.py
```

### Step 3: Review Outputs

```bash
# View markdown summary
cat reports/EXPERIMENTAL_SUMMARY.md

# Check LaTeX tables
cat reports/latex_tables.tex

# Verify all files generated
ls -R reports/
```

### Step 4: Use in Your Report

**For LaTeX reports**:
```latex
\documentclass{article}
\usepackage{booktabs}

\begin{document}

\section{Experimental Results}

\input{reports/latex_tables.tex}

% Reference specific tables
As shown in Table~\ref{tab:attack_eval}, the model exhibits severe vulnerability...

\end{document}
```

**For Markdown reports**:
Simply copy sections from `EXPERIMENTAL_SUMMARY.md` into your documentation.

**For presentations**:
Use the figure paths from Section 6 to include visualizations.

---

## Customization

### Modify Tables

Edit the generator methods in `utils/report_generator.py`:

```python
def _generate_attack_eval_table(self):
    # Customize table format, columns, or content
    latex = []
    latex.append("\\begin{table}[h]")
    # Your custom table code
    return "\n".join(latex)
```

### Add New Sections

Extend the markdown summary:

```python
def generate_markdown_summary(self):
    md = []
    # ... existing sections ...

    # Add your custom section
    md.append("\n## 7. My Custom Section\n")
    md.append(self._my_custom_section())

    return "\n".join(md)
```

### Filter Results

Load only specific results:

```python
generator = ReportGenerator()
generator.load_all_results()

# Access specific data
if generator.attack_results:
    fgsm_data = generator.attack_results['fgsm']
    # Process as needed
```

---

## Troubleshooting

### Missing Results

**Problem**: "Attack evaluation results not found"

**Solution**: Run the corresponding experiment first:
```bash
python attacks/evaluate_attacks.py
```

### Incomplete Tables

**Problem**: Some tables are empty or have missing data

**Solution**: Check that all experiments completed successfully and saved results to the expected paths.

### Figure Paths Incorrect

**Problem**: Figure paths in markdown don't match actual locations

**Solution**: Verify the results directory structure matches expectations:
```
results/
├── attack_evaluation/
├── baseline_training/
├── hyperparameter_analysis/
└── visualizations/
```

### LaTeX Compilation Errors

**Problem**: Tables don't compile in LaTeX

**Solution**:
1. Ensure `\usepackage{booktabs}` is in your preamble
2. Check for special characters that need escaping
3. Verify table syntax with a simple test document

---

## Advanced Usage

### Programmatic Access

```python
from utils.report_generator import ReportGenerator

# Create generator
gen = ReportGenerator()
gen.load_all_results()

# Access raw data
if gen.attack_results:
    clean_acc = gen.attack_results['clean']['accuracy']
    print(f"Clean accuracy: {clean_acc * 100:.2f}%")

# Generate only specific outputs
latex_output = gen.generate_latex_tables()
with open('my_tables.tex', 'w') as f:
    f.write(latex_output)

md_output = gen.generate_markdown_summary()
with open('my_summary.md', 'w') as f:
    f.write(md_output)
```

### Batch Processing

Generate reports for multiple experimental runs:

```python
import os
from utils.report_generator import ReportGenerator

experiment_dirs = ['exp1', 'exp2', 'exp3']

for exp_dir in experiment_dirs:
    gen = ReportGenerator(
        results_dir=f'./results_{exp_dir}',
        output_dir=f'./reports_{exp_dir}'
    )
    gen.generate_all_reports()
```

### Custom Formatting

Override formatting methods:

```python
class CustomReportGenerator(ReportGenerator):
    def _generate_attack_eval_table(self):
        # Your custom table format
        latex = []
        # ... custom implementation ...
        return "\n".join(latex)

# Use custom generator
gen = CustomReportGenerator()
gen.generate_all_reports()
```

---

## Output Structure

After running, your reports directory will contain:

```
reports/
├── latex_tables.tex              # All LaTeX tables
├── EXPERIMENTAL_SUMMARY.md       # Complete markdown summary
└── (optionally) custom_report.md # If you add custom reports
```

Additionally, referenced files:

```
results/
├── attack_evaluation/
│   ├── attack_results.json
│   ├── attack_results_summary.txt
│   ├── attack_comparison.png
│   └── attack_visual_grid.png
├── baseline_training/
│   ├── baseline_metrics.json
│   ├── baseline_training_curves.png
│   └── ...
├── hyperparameter_analysis/
│   ├── results.json
│   ├── epsilon_analysis.png
│   └── ...
└── visualizations/
    ├── fgsm_comparison.png
    ├── confusion_matrix_clean.png
    └── ...
```

---

## Best Practices

### 1. Version Control
- Commit generated reports with experiments
- Tag releases with corresponding reports
- Include generation timestamp in reports

### 2. Documentation
- Document any custom modifications
- Keep a changelog of report format changes
- Note which figures correspond to which experiments

### 3. Reproducibility
- Include experimental parameters in report metadata
- Document random seeds and configurations
- Archive all result JSON files

### 4. Presentation
- Use high-DPI figures (300+ DPI) for publications
- Verify all tables compile correctly in LaTeX
- Test markdown rendering on target platform

### 5. Validation
- Cross-check numbers between JSON and reports
- Verify figure paths are correct
- Ensure all experiments completed successfully

---

## Integration with Paper Writing

### For Conference Papers

```latex
\documentclass[conference]{IEEEtran}
\usepackage{booktabs}

\begin{document}

\section{Experimental Results}

We evaluated our baseline ResNet18 model against FGSM and PGD
attacks on the CIFAR-10 dataset. As shown in
Table~\ref{tab:attack_eval}, the model demonstrates severe
vulnerability to adversarial perturbations.

\input{reports/latex_tables.tex}

The epsilon analysis (Figure~\ref{fig:epsilon}) reveals a critical
threshold at $\epsilon \approx 0.03$...

\begin{figure}[h]
\centering
\includegraphics[width=0.48\textwidth]{results/hyperparameter_analysis/epsilon_analysis.png}
\caption{Attack effectiveness vs epsilon value}
\label{fig:epsilon}
\end{figure}

\end{document}
```

### For Journal Articles

Use the detailed metrics from the markdown summary and include comprehensive tables and figures from all analyses.

### For Technical Reports

Include the entire EXPERIMENTAL_SUMMARY.md as an appendix, with LaTeX tables in the main body.

---

## FAQ

**Q: Can I generate reports without running all experiments?**
A: Yes, the generator will work with whatever results are available and note missing data.

**Q: How do I add custom metrics to the reports?**
A: Extend the `ReportGenerator` class and override the relevant generation methods.

**Q: Can I export to formats other than LaTeX and Markdown?**
A: Yes, you can add methods to generate HTML, PDF, or other formats using the loaded data.

**Q: How do I update reports after re-running experiments?**
A: Simply re-run `python utils/report_generator.py` - it will overwrite with latest results.

**Q: Can I generate reports for a subset of experiments?**
A: Yes, specify custom paths or modify the `load_all_results()` method to load only specific files.

---

**Ready to generate?** Run `python utils/report_generator.py` and get publication-ready outputs!
