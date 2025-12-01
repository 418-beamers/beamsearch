import torch
from typing import Optional, Tuple

try:
    import beamsearch_cuda_native
except ImportError:
    raise ImportError(
        "beamsearch_cuda_native not found. Please build the extension first."
    )


class CTCBeamSearchDecoder:
    """
    CUDA-accelerated CTC Beam Search Decoder

    This decoder implements the standard CTC beam search algorithm as described in:
    "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with
    Recurrent Neural Networks" (Graves et al., 2006)

    Args:
        beam_width: Number of beams to maintain during search
        num_classes: Size of vocabulary (including blank token)
        max_output_length: Maximum length of output sequences
        blank_id: Index of the blank token (typically 0)
        batch_size: Number of sequences to process in parallel
        max_time: Maximum number of time steps in input
    """

    def __init__(
        self,
        beam_width: int,
        num_classes: int,
        max_output_length: int,
        blank_id: int = 0,
        batch_size: int = 1,
        max_time: int = 100
    ):
        self.beam_width = beam_width
        self.num_classes = num_classes
        self.max_output_length = max_output_length
        self.blank_id = blank_id
        self.batch_size = batch_size
        self.max_time = max_time

        self.state_ptr = beamsearch_cuda_native.create_ctc_beam_search_state(
            batch_size,
            beam_width,
            num_classes,
            max_time,
            max_output_length,
            blank_id
        )

    def __del__(self):
        if hasattr(self, 'state_ptr'):
            beamsearch_cuda_native.free_ctc_beam_search_state(self.state_ptr)

    def decode(
        self,
        log_probs: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode CTC log probabilities using beam search

        Args:
            log_probs: Log probabilities from CTC layer
                      Shape: [batch_size, max_time, num_classes]
                      Should be on CUDA device and dtype float32
            input_lengths: Optional tensor of actual sequence lengths
                          Shape: [batch_size]
                          If None, assumes all sequences are max_time length

        Returns:
            sequences: Decoded sequences
                      Shape: [batch_size, beam_width, max_output_length]
                      Values: token indices (-1 for padding)
            lengths: Length of each decoded sequence
                    Shape: [batch_size, beam_width]
            scores: Log probability scores for each beam
                   Shape: [batch_size, beam_width]

        Example:
            >>> decoder = CTCBeamSearchDecoder(
            ...     beam_width=10,
            ...     num_classes=29,  # 28 characters + blank
            ...     max_output_length=50,
            ...     blank_id=0,
            ...     batch_size=4,
            ...     max_time=100
            ... )
            >>> log_probs = torch.randn(4, 100, 29).cuda()
            >>> sequences, lengths, scores = decoder.decode(log_probs)
            >>> best_sequence = sequences[:, 0, :]  # Best beam for each batch
        """
        if not log_probs.is_cuda:
            raise ValueError("log_probs must be on CUDA device")
        if log_probs.dtype != torch.float32:
            raise ValueError("log_probs must be float32")
        if log_probs.dim() != 3:
            raise ValueError(
                f"log_probs must be 3D [batch_size, max_time, num_classes], "
                f"got shape {log_probs.shape}"
            )

        batch_size, max_time, num_classes = log_probs.shape

        if batch_size != self.batch_size:
            raise ValueError(
                f"batch_size mismatch: expected {self.batch_size}, got {batch_size}"
            )
        if max_time != self.max_time:
            raise ValueError(
                f"max_time mismatch: expected {self.max_time}, got {max_time}"
            )
        if num_classes != self.num_classes:
            raise ValueError(
                f"num_classes mismatch: expected {self.num_classes}, got {num_classes}"
            )

        if input_lengths is not None:
            if not input_lengths.is_cuda:
                raise ValueError("input_lengths must be on CUDA device")
            if input_lengths.dtype != torch.int32:
                input_lengths = input_lengths.to(torch.int32)
            if input_lengths.shape != (batch_size,):
                raise ValueError(
                    f"input_lengths must have shape [{batch_size}], "
                    f"got {input_lengths.shape}"
                )
        else:
            input_lengths = torch.empty(0, dtype=torch.int32, device='cuda')

        beamsearch_cuda_native.initialize_ctc_beam_search(
            self.state_ptr,
            self.batch_size,
            self.beam_width,
            self.num_classes,
            self.max_time,
            self.max_output_length,
            self.blank_id
        )

        beamsearch_cuda_native.run_ctc_beam_search(
            self.state_ptr,
            log_probs,
            self.batch_size,
            self.beam_width,
            self.num_classes,
            self.max_time,
            self.max_output_length,
            self.blank_id,
            input_lengths
        )

        sequences = beamsearch_cuda_native.get_sequences(
            self.state_ptr,
            self.batch_size,
            self.beam_width,
            self.max_output_length
        )

        lengths = beamsearch_cuda_native.get_sequence_lengths(
            self.state_ptr,
            self.batch_size,
            self.beam_width
        )

        scores = beamsearch_cuda_native.get_scores(
            self.state_ptr,
            self.batch_size,
            self.beam_width
        )

        return sequences, lengths, scores

    def decode_greedy(
        self,
        log_probs: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode using only the best beam (greedy decoding)

        This is a convenience method that runs beam search and returns only
        the best (highest scoring) sequence for each batch element.

        Args:
            log_probs: Log probabilities from CTC layer
                      Shape: [batch_size, max_time, num_classes]
            input_lengths: Optional tensor of actual sequence lengths
                          Shape: [batch_size]

        Returns:
            sequences: Best decoded sequence for each batch
                      Shape: [batch_size, max_output_length]
            lengths: Length of each decoded sequence
                    Shape: [batch_size]
        """
        sequences, lengths, scores = self.decode(log_probs, input_lengths)

        best_sequences = sequences[:, 0, :]
        best_lengths = lengths[:, 0]

        return best_sequences, best_lengths


def ctc_beam_search_decode(
    log_probs: torch.Tensor,
    beam_width: int = 10,
    blank_id: int = 0,
    input_lengths: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience function for one-shot CTC beam search decoding

    This function automatically creates a decoder instance and runs decoding.
    For repeated decoding with the same parameters, use CTCBeamSearchDecoder directly
    to avoid repeated state allocation.

    Args:
        log_probs: Log probabilities from CTC layer
                  Shape: [batch_size, max_time, num_classes]
        beam_width: Number of beams to maintain
        blank_id: Index of blank token
        input_lengths: Optional actual sequence lengths

    Returns:
        sequences: Shape [batch_size, beam_width, max_output_length]
        lengths: Shape [batch_size, beam_width]
        scores: Shape [batch_size, beam_width]
    """
    batch_size, max_time, num_classes = log_probs.shape
    max_output_length = max_time  

    decoder = CTCBeamSearchDecoder(
        beam_width=beam_width,
        num_classes=num_classes,
        max_output_length=max_output_length,
        blank_id=blank_id,
        batch_size=batch_size,
        max_time=max_time
    )

    return decoder.decode(log_probs, input_lengths)

__all__ = ['CTCBeamSearchDecoder', 'ctc_beam_search_decode']