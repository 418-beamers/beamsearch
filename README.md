# Setup
Run the following commands to create a conda env ready to run the test harness (temporarily for CPU only)
```bash
conda create -n beams python=3.10 -y
conda activate beams
python -m pip install --upgrade pip setuptools wheel ninja
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install flashlight-text
```

Might also have to run these to add CUDA toolkit to path: 
```
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

can run `python testing/env_test.py` to validate that the basic libraries necessary for the project are present

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

### Decoder Test Arguments

The `testing/ctc_decoder_test.py` script accepts several arguments to control the test parameters:

- `--batch-size`: Batch size for the input (default: 2).
- `--time-steps`: Length of the input sequences (default: 120).
- `--vocab-size`: Size of the vocabulary (default: 32).
- `--beam-width`: Width of the beam for the search (default: 50).
- `--top-k`: Number of top hypotheses to return (default: 3).
- `--candidate-device`: Device to run the candidate decoder on (`cuda`). If not specified, only the reference decoder runs (on CPU).
- `--seed`: Manual seed for reproducibility (default: 0).

### Example Usage

Run with custom parameters:
```bash
python testing/ctc_decoder_test.py --batch-size 4 --time-steps 150 --vocab-size 32 --beam-width 20 --candidate-device cuda
```

Run only the reference decoder:
```bash
python testing/ctc_decoder_test.py --batch-size 2
```

# Running the tests 
Install the extension:

```bash
pip install -e beamsearch_cuda/
```

Run the tests:
```bash
python testing/ctc_decoder_test.py --candidate-device cuda
```

# CTC Beam Search Usage

`beamsearch_cuda.beam_search.ctc_beam_search` runs the CUDA decoder on tensors passed for 
`log_probs` and `input_lengths`. 

Example usage:

```python
import torch
from beamsearch_cuda import beam_search

log_probs = torch.randn(2, 8, 32, device="cuda").log_softmax(-1)
lengths = torch.tensor([8, 6], dtype=torch.int32, device="cuda")

beam_search.ctc_beam_search(
    log_probs=log_probs,
    input_lengths=lengths,
    beam_width=4,
    blank_idx=0,
    top_k=2,
)
```
