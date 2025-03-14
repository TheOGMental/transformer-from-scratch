# Transformers from Scratch

This repository contains our group's implementation of a transformer model from scratch, built as part of the **MATH598B** course at *Colorado School of Mines*, Spring 2025. The project was completed by March 16, 2025, and includes a fully functional transformer with attention heads, MLPs, RMS normalization, a pretrained tokenizer, positional embeddings, and a basic generation method. We trained the model on a dataset of Gutenberg books and plotted the training loss to analyze its behavior.

## Team Members and Contributions

### Human Contributions

#### Griffin Rutherford:
- Initialized the project repository and drafted the initial transformer architecture in `T1000.py` by February 26, 2025.
- Implemented RMS normalization (RMSNorm) in `T1000.py`, integrating it into the transformer with a pre-norm configuration, completed by March 13, 2025.
- Debugged dimensional mismatches in the attention head to ensure compatibility with tests, finalized by February 26, 2025.
- Created the `llm-transcripts/` directory and uploaded his LLM chat log as `Griffin-Rutherford-LLM-Transcript.pdf` on March 14, 2025.
- Ensured task alignment and test passage via Discord by March 14, 2025.

#### Matthew Stanley (Triblades):
- Converted initial test notebooks into a robust unittest suite in `tests/test.py`, completed by February 26, 2025.
- Configured GitHub Actions to automatically run tests on push requests, enhancing CI/CD, implemented by February 26, 2025.
- Integrated the tiktoken tokenizer into `main.py` and `utility.py` for efficient text processing, completed by March 14, 2025.
- Diagnosed numerical instability (e.g., NaN outputs) in early transformer runs, suggesting overflow fixes by February 26, 2025.
- Collaborated on test refinements, ensuring all components passed after architectural changes, finalized by March 14, 2025.

#### Gabriel Del Castillo (Melon):
- Set up a GitHub webhook to notify the team of repository updates, completed by March 11, 2025.
- Implemented the `.generate()` method in `T1000.py` for text generation from prompts, finalized by March 15, 2025.
- Defined the project timeline, setting the deadline of March 16, 2025, and managed task discussions in class.
- Contributed to planning remaining features (e.g., tokenization, embeddings) via Discord, completed by March 11, 2025.
- Ensured team alignment on submission requirements, including LLM documentation, finalized by March 14, 2025.

#### Alex Fruge (st0rm):
- Shared the repository link with instructor Michael via email, ensuring visibility, completed by February 26, 2025.
- Added positional embeddings to the Transformer class in `T1000.py`, enhancing sequence awareness, completed by March 15, 2025.
- Validated test failures across environments, aiding early debugging efforts, noted by February 26, 2025.
- Contributed to the training setup in `main.py`, ensuring compatibility with the updated model, finalized by March 14, 2025.
- Supported team communication by confirming task feasibility (e.g., embeddings timeline), completed by March 14, 2025.

### LLM Contributions (Grok 3 by xAI)

#### Assisted Team Members:
- Provided detailed code and explanations for RMSNorm implementation, used from March 12–14, 2025.
- Guided architectural modifications (pre-norm, residual restructuring), completed by March 13, 2025.
- Helped update `tests/test.py` to pass with the new architecture, finalized by March 14, 2025.
- Generated this README draft, summarizing contributions and structure, on March 14, 2025.
- Explained RMSNorm conceptually, aiding Griffin's implementation decisions, used by March 13, 2025.

**Usage Details:**
- *Most Useful*: Code generation and normalization theory.
- *Least Useful*: Required human verification for test-specific fixes.
- *Transcript*: `llm-transcripts/Griffin-Rutherford-LLM-Transcript.pdf`.

## Project Structure

```
transformer-from-scratch/
├── README.md              # Project overview and instructions (this file)
├── T1000.py              # Core transformer implementation (Transformer, AttentionHead, MLP, RMSNorm)
├── __pycache__/          # Compiled Python files (auto-generated)
│   ├── T1000.cpython-313.pyc
│   ├── utility.cpython-310.pyc
│   └── utility.cpython-313.pyc
├── batch_loss_values.png  # Plot of training loss from main.py
├── llm-transcripts/      # Directory for LLM chat transcripts
│   └── Griffin-Rutherford-LLM-Transcript.pdf
├── main.ipynb            # Jupyter notebook for initial experimentation
├── main.py               # Training script with dataset loading and loss plotting
├── requirements.txt      # List of Python dependencies
├── tests/                # Test suite directory
│   ├── __pycache__/      # Compiled test files
│   │   └── test.cpython-313.pyc
│   └── test.py           # Unit tests for transformer components
└── utility.py            # Helper functions (data loading, tokenization)
```

**Where to Look for Code:**
- Transformer implementation: `T1000.py`
- Training and dataset handling: `main.py`
- Tests: `tests/test.py`
- Utilities (e.g., Gutenberg data loader, tokenizer): `utility.py`

## Dependencies

Dependencies are listed in `requirements.txt`. Key packages include:

- **torch==2.6.0**: Tensor operations and model implementation.
- **tiktoken==0.2.0**: Pretrained tokenization.
- **matplotlib==3.10.0**: Loss plotting.
- **unittest**: Built-in, for tests.

*Full list*: See `requirements.txt`.

To install:

```bash
pip install -r requirements.txt
```

## Virtual Environment Setup

We used virtual environments for dependency management across platforms.

### MacOS (Griffin's Setup)
**Creation:**
```bash
python3 -m venv venvgriffinrutherford
```
**Activation:**
```bash
source venvgriffinrutherford/bin/activate
```
*Notes*: Python 3.13, per `__pycache__/*.cpython-313.pyc`.

### Linux (Others' Setup)
**Creation:**
```bash
python3 -m venv venv
```
**Activation:**
```bash
source venv/bin/activate
```
*Notes*: Some used Python 3.10 (e.g., `utility.cpython-310.pyc`), but tests passed across versions.

## General Instructions

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd transformer-from-scratch
   ```
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run tests or training scripts.

## Running the Project

**Tests:**
```bash
python -m unittest tests/test.py -v
```

**Training:**
```bash
python main.py
```
Trains on Gutenberg books, saves `batch_loss_values.png` and `model.pt`.

**Requirements Met:**
- Includes attention heads, MLPs, RMSNorm, positional embeddings, tiktoken tokenizer, and `.generate()`.

## Loss Analysis

The loss plot (`batch_loss_values.png`) showed a decreasing trend, stabilized by RMSNorm. Initial fluctuations reduced over time, though the batch size of 1 and single epoch limited full convergence.

---

**Submitted By**: Griffin Rutherford, Matthew Stanley, Gabriel Del Castillo, Alex Fruge

**Course**: MATH598B, Spring 2025

**Instructor**: Michael Ivanitskiy

**Deadline**: March 16, 2025
