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

## Simple decoder testing harness

Use `testing/ctc_decoder_test.py` to run the flashlight-based (https://arxiv.org/pdf/2201.12465) reference decoder and our development decoder on randomly generated log-probabilities. 

```bash
python testing/ctc_decoder_test.py --batch-size 4 --time-steps 150 --vocab-size 32
```
To compare against CUDA implementation:

```bash
python testing/ctc_decoder_test.py --candidate-device cuda
```

# CTC CUDA hello world

`beamsearch_cuda.beam_search.ctc_beam_search` currently loads a lightweight CUDA
launches a simple hello world kernel. Run it by passing CUDA tensors for both 
`log_probs` and `input_lengths`. 

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

*This is just a first test to make sure the dev environment works*
