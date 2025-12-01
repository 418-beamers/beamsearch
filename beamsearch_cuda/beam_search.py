from typing import Tuple
import torch

"""
implement the torch autograd function here 
"""

def ctc_beam_search(
    log_probs: torch.Tensor, 
    input_lengths: torch.Tensor, 
    beam_width: int = 1, # default to greedy decoding 
    blank_idx: int = 0, # default first in vocab
    top_k: int = 1 # default to return top-1 per batch element
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched CTC Beam Search decoder (python interface)

    args: 
        log_probs: 
            tensor of shape (batch_size, time_steps, vocab_size), containing 
            log probabilities (i.e. log softmax output)
            **should be on device for CUDA decoding**

        input_lengths: 
            int tensor of shape (batch_size,) containing correct length of each 
            sequence in log_probs 

        beam_width: 
            int, number of hypotheses kept per time step 

        blank_idx: 
            int, index of CTC blank symbol 
        
        top_k:
            int, number of hypotheses returned per element in batch

    returns:  
        hypotheses: 
            tensor of shape (batch_size, top_k, max_decoded_length), with padding for 
            hypotheses shorter than max_decoded_length 

        scores:
            tensor of shape (batch_size, top_k), containing log probability scores for each 
            returned hypothesis
    """

    raise NotImplementedError