# Interface
Let: 
- B: batch_size, 
- T: time_steps, 
- V: vocab_size,

*ctc_beam_search*
- args:
    - log_probs: tensor, (B, T, V)
    - input_lengths: tensor, (B,)
    - beam_width: int 
    - blank_idx: int
    - top_k: int

- returns:
    - hypotheses: tensor, (B, top_k, max_decoded_length)
    - scores: tensor, (B, top_k)

## A note on standard CTC Beam search decoding vs. Auto-regressive beam search decoding
This projects primary focus is on standard CTC Beam search decoding, as it is significantly more amenable to parallelization with CUDA. In this approach, the neural network outputs the complete distribution for all tokens over all timesteps in a forward pass, B x T x V, as described above. The algorithm operates on these pre-computed log-probabilities to perform top-K selection. With this approach, a subsequently generated token does not depend on a previously generated one, simplifying the problem mathematically and offering more axes to explore parallelization over. 

By contrast, auto-regressive Beam search has a dependency between previously and subsequently generated tokens, creating a sequential dependency which requires repeated queries of the neural network's forward pass. 

In practice, CTC Beam search decoding is preferred for ASR (automatic speech recognition), especially in streaming/real-time applications, whereas auto-regressive Beam search is preferred for language models where the token dependencies matter more.

# Setup
Run the following commands to create a conda env ready to run the test harness
```bash
conda create -n beams python=3.10 -y
conda activate beams
python -m pip install --upgrade pip setuptools wheel ninja
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install flashlight-text rich soundfile
```

Might also have to run these to add CUDA toolkit to path: 
```
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="7.5"
```

can run `python testing/env_test.py` to validate that the basic libraries necessary for the project are present

# Quickstart

## Building CUDA Extension 

Install the extension by running:
```bash
pip install -e beamsearch_cuda/ --no-build-isolation
```

Verify CUDA toolchain with hello-world extension:
```bash
python testing/ctc_decoder_test.py --hello
```

Run synthetic benchmark:
```bash
python testing/ctc_decoder_test.py --candidate-device cuda
```

Run real audio test:
```bash
python testing/ctc_decoder_test.py --real --candidate-device cuda --verbose
```

# Testing Module

The `testing/ctc_decoder_test.py` script accepts several arguments to control the test parameters:

**Input Generation:**
- `--batch-size`: Batch size for the input (default: 2)
- `--time-steps`: Length of the input sequences (default: 120)
- `--vocab-size`: Size of the vocabulary (default: 32)
- `--seed`: Manual seed for reproducibility (default: 0)

**Decoder Options:**
- `--beam-width`: Width of the beam for the search (default: 50)
- `--top-k`: Number of top hypotheses to return (default: 3)
- `--candidate-device`: Device to run the candidate decoder on (`cuda`). If not specified, only the reference decoder runs (on CPU)

**Timing:**
- `--timing-runs`: Number of timing repetitions per decoder (default: 3)

**Real Audio Mode:**
- `--real`: Use real audio with pre-trained Wav2Vec2 ASR model instead of synthetic data
- `--audio-file`: Path to audio file for real mode (if not provided, downloads a sample)

**Adaptive Beam Width Scheduling:**
- `--adaptive-beam-width`: Enable adaptive beam width scheduling (beam width decays over time)
- `--scheduler-type`: Scheduler type: `naive`, `lut`, or `mlp` (default: `naive`)
  - `naive`: Uses exponential decay formula `w(t) = a * exp(-b * t) + c`
  - `lut`: Uses pre-computed lookup table from binary file
  - `mlp`: Uses trained MLP model to predict decay parameters
- `--schedule-a`: Decay parameter A for naive scheduler (amplitude)
- `--schedule-b`: Decay parameter B for naive scheduler (decay rate)
- `--schedule-c`: Decay parameter C for naive scheduler (asymptote)
- `--schedule-min`: Minimum beam width
- `--schedule-init`: Initial beam width before decay
- `--schedule-init-steps`: Number of timesteps to hold initial width before decay
- `--lut-path`: Path to LUT scheduler binary (default: `testing/bin/scheduler_lut.bin`, auto-generated if missing)
- `--mlp-path`: Path to MLP scheduler weights (default: `testing/bin/mlp_weights.bin`, auto-copied if missing)

**Other:**
- `--verbose`: Print decoded sequences and similarity metrics
- `--hello`: Run hello-world CUDA extension for toolchain verification

### Example Usage

**Synthetic Data (Default)**

Run with custom parameters:
```bash
python testing/ctc_decoder_test.py --batch-size 4 --time-steps 150 --vocab-size 32 --beam-width 20 --candidate-device cuda
```

Run only the reference decoder:
```bash
python testing/ctc_decoder_test.py --batch-size 2
```

**Real Audio Mode**

Run with a sample .wav (audio) file (auto-downloaded):
```bash
python testing/ctc_decoder_test.py --real --candidate-device cuda
```

or with your own .wav file
```bash
python testing/ctc_decoder_test.py --real --audio-file ~/beamsearch/testing/beamers_sample.wav --candidate-device cuda
```

Real audio mode uses the Wav2Vec2 ASR model from torchaudio to generate CTC log probabilities from actual speech.

**Adaptive Beam Width Scheduling**

Run with naive exponential decay (start at 50, decay to 10):
```bash
python testing/ctc_decoder_test.py --real --candidate-device cuda \
    --adaptive-beam-width --scheduler-type naive \
    --schedule-a 40 --schedule-b 0.1 --schedule-c 10 \
    --schedule-init 50 --schedule-min 10 --schedule-init-steps 5
```

Run with LUT scheduler (auto-generates `testing/bin/scheduler_lut.bin` if missing):
```bash
python testing/ctc_decoder_test.py --real --candidate-device cuda \
    --adaptive-beam-width --scheduler-type lut \
    --schedule-init 50 --schedule-min 10
```

Run with MLP scheduler (auto-copies weights to `testing/bin/mlp_weights.bin` if missing):
```bash
python testing/ctc_decoder_test.py --real --candidate-device cuda \
    --adaptive-beam-width --scheduler-type mlp \
    --schedule-init 50 --schedule-min 10
```

## Test Module Structure

The testing harness is organized into modular components under `testing/utils/`:

```
testing/
├── ctc_decoder_test.py    # main test script
├── env_test.py            # environment validation
└── utils/
    ├── __init__.py        # module exports
    ├── timing.py          # benchmarking utilities (run_timed, print_timing_table)
    ├── similarity.py      # distance metrics (levenshtein, edit distance)
    ├── tokenization.py    # vocab handling (make_tokens, detokenize, formatting)
    ├── loaders.py         # extension loading (load_candidate_module, load_hello_extension)
    ├── inputs.py          # synthetic input generation
    └── real_audio.py      # Wav2Vec2 audio processing for real ASR testing
```
