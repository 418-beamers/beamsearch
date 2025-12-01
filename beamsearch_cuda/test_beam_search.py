import torch
from beam_search import CTCBeamSearchDecoder, ctc_beam_search_decode


def main():
    print("=" * 80)
    print("CTC Beam Search Decoder - Example Usage and Tests")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This decoder requires a CUDA-enabled GPU.")
        exit(1)

    print(f"\nUsing device: {torch.cuda.get_device_name(0)}")
    print()

    # =========================================================================
    # Example 1: Basic usage with random inputs
    # =========================================================================
    print("Example 1: Basic usage with random log probabilities")
    print("-" * 80)

    batch_size = 2
    max_time = 50
    num_classes = 29  # 28 characters (a-z, space, apostrophe) + blank
    beam_width = 5

    decoder = CTCBeamSearchDecoder(
        beam_width=beam_width,
        num_classes=num_classes,
        max_output_length=max_time,
        blank_id=0,
        batch_size=batch_size,
        max_time=max_time
    )

    log_probs = torch.randn(batch_size, max_time, num_classes).cuda()
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)

    sequences, lengths, scores = decoder.decode(log_probs)

    print(f"Input shape: {log_probs.shape}")
    print(f"Output sequences shape: {sequences.shape}")
    print(f"Output lengths shape: {lengths.shape}")
    print(f"Output scores shape: {scores.shape}")
    print()

    for i in range(batch_size):
        print(f"Batch {i}:")
        for j in range(min(3, beam_width)):  
            seq = sequences[i, j, :lengths[i, j]].cpu().numpy()
            score = scores[i, j].item()
            print(f"  Beam {j}: length={lengths[i, j].item()}, score={score:.4f}")
            print(f"           tokens={seq}")
        print()

    # =========================================================================
    # Example 2: Using the convenience function
    # =========================================================================
    print("\nExample 2: Using the convenience function ctc_beam_search_decode()")
    print("-" * 80)

    log_probs_2 = torch.randn(1, 30, 10).cuda() 
    log_probs_2 = torch.nn.functional.log_softmax(log_probs_2, dim=-1)

    sequences_2, lengths_2, scores_2 = ctc_beam_search_decode(
        log_probs_2,
        beam_width=3,
        blank_id=0
    )

    print(f"Input shape: {log_probs_2.shape}")
    print(f"Best sequence: {sequences_2[0, 0, :lengths_2[0, 0]].cpu().numpy()}")
    print(f"Best score: {scores_2[0, 0].item():.4f}")
    print()

    # =========================================================================
    # Example 3: Using input_lengths for variable-length sequences
    # =========================================================================
    print("\nExample 3: Variable-length sequences with input_lengths")
    print("-" * 80)

    batch_size_3 = 3
    max_time_3 = 40
    num_classes_3 = 15

    decoder_3 = CTCBeamSearchDecoder(
        beam_width=4,
        num_classes=num_classes_3,
        max_output_length=max_time_3,
        blank_id=0,
        batch_size=batch_size_3,
        max_time=max_time_3
    )

    log_probs_3 = torch.randn(batch_size_3, max_time_3, num_classes_3).cuda()
    log_probs_3 = torch.nn.functional.log_softmax(log_probs_3, dim=-1)

    input_lengths = torch.tensor([40, 25, 30], dtype=torch.int32).cuda()

    sequences_3, lengths_3, scores_3 = decoder_3.decode(log_probs_3, input_lengths)

    print(f"Input shape: {log_probs_3.shape}")
    print(f"Input lengths: {input_lengths.cpu().numpy()}")
    print()

    for i in range(batch_size_3):
        print(f"Batch {i} (actual length: {input_lengths[i].item()}):")
        best_seq = sequences_3[i, 0, :lengths_3[i, 0]].cpu().numpy()
        best_score = scores_3[i, 0].item()
        print(f"  Best beam: length={lengths_3[i, 0].item()}, score={best_score:.4f}")
        print(f"             tokens={best_seq}")
        print()

    # =========================================================================
    # Example 4: Greedy decoding (best beam only)
    # =========================================================================
    print("\nExample 4: Greedy decoding (best beam only)")
    print("-" * 80)

    log_probs_4 = torch.randn(2, 25, 10).cuda()
    log_probs_4 = torch.nn.functional.log_softmax(log_probs_4, dim=-1)

    decoder_4 = CTCBeamSearchDecoder(
        beam_width=5,
        num_classes=10,
        max_output_length=25,
        blank_id=0,
        batch_size=2,
        max_time=25
    )

    best_sequences, best_lengths = decoder_4.decode_greedy(log_probs_4)

    print(f"Input shape: {log_probs_4.shape}")
    print(f"Best sequences shape: {best_sequences.shape}")
    print()

    for i in range(2):
        seq = best_sequences[i, :best_lengths[i]].cpu().numpy()
        print(f"Batch {i}: length={best_lengths[i].item()}")
        print(f"           tokens={seq}")
        print()

    # =========================================================================
    # Example 5: Realistic example with character vocabulary
    # =========================================================================
    print("\nExample 5: Realistic example with character mapping")
    print("-" * 80)


    vocab = ['<blank>'] + list('abcdefghijklmnopqrstuvwxyz') + [' ']
    idx_to_char = {i: c for i, c in enumerate(vocab)}

    batch_size_5 = 1
    max_time_5 = 20
    num_classes_5 = 28  

    decoder_5 = CTCBeamSearchDecoder(
        beam_width=10,
        num_classes=num_classes_5,
        max_output_length=max_time_5,
        blank_id=0,
        batch_size=batch_size_5,
        max_time=max_time_5
    )

    # Create biased probabilities that favor spelling "hello"
    # h=8, e=5, l=12, o=15
    log_probs_5 = torch.full((1, max_time_5, num_classes_5), -10.0).cuda()

    # Time steps where we want each character
    # We'll spike probabilities at specific time steps to guide the decoder
    log_probs_5[0, 2, 8] = -0.1   # 'h' at time 2
    log_probs_5[0, 5, 5] = -0.1   # 'e' at time 5
    log_probs_5[0, 8, 12] = -0.1  # 'l' at time 8
    log_probs_5[0, 11, 12] = -0.1 # 'l' at time 11
    log_probs_5[0, 14, 15] = -0.1 # 'o' at time 14

    # Set blank probabilities for other time steps
    for t in [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19]:
        log_probs_5[0, t, 0] = -0.1  # blank

    sequences_5, lengths_5, scores_5 = decoder_5.decode(log_probs_5)

    print(f"Vocabulary: {vocab}")
    print("Attempting to decode a sequence...")
    print()

    # Show top 3 beams with character translation
    for j in range(min(3, 10)):
        seq_indices = sequences_5[0, j, :lengths_5[0, j]].cpu().numpy()
        decoded_text = ''.join([idx_to_char.get(int(idx), '?') for idx in seq_indices])
        score = scores_5[0, j].item()
        print(f"Beam {j}: \"{decoded_text}\" (score={score:.4f})")

    print()

    # =========================================================================
    # Example 6: Stress test with larger inputs
    # =========================================================================
    print("\nExample 6: Stress test with larger batch and longer sequences")
    print("-" * 80)

    batch_size_6 = 8
    max_time_6 = 100
    num_classes_6 = 50
    beam_width_6 = 20

    decoder_6 = CTCBeamSearchDecoder(
        beam_width=beam_width_6,
        num_classes=num_classes_6,
        max_output_length=max_time_6,
        blank_id=0,
        batch_size=batch_size_6,
        max_time=max_time_6
    )

    log_probs_6 = torch.randn(batch_size_6, max_time_6, num_classes_6).cuda()
    log_probs_6 = torch.nn.functional.log_softmax(log_probs_6, dim=-1)

    print(f"Processing large input: {log_probs_6.shape}")
    print(f"Beam width: {beam_width_6}")

    import time
    start = time.time()
    sequences_6, lengths_6, scores_6 = decoder_6.decode(log_probs_6)
    elapsed = time.time() - start

    print(f"Decoding completed in {elapsed:.4f} seconds")
    print(f"Output shape: {sequences_6.shape}")
    print(f"Average sequence length: {lengths_6[:, 0].float().mean().item():.2f}")
    print(f"Best scores: {scores_6[:, 0].cpu().numpy()}")
    print()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
