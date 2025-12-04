"""
distance metrics for comparison
"""

from __future__ import annotations

# implementation basedon : https://www.geeksforgeeks.org/dsa/introduction-to-levenshtein-distance/
def levenshtein_distance(tokens_a: list, tokens_b: list) -> int:

    if len(tokens_a) < len(tokens_b):
        tokens_a, tokens_b = tokens_b, tokens_a

    previous_row = list(range(len(tokens_b) + 1))

    for i, token_a in enumerate(tokens_a, start=1):
        current_row = [i]

        for j, token_b in enumerate(tokens_b, start=1):
            cost = 0 if token_a == token_b else 1
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[j] + 1,
                    previous_row[j - 1] + cost,
                )
            )

        previous_row = current_row

    return previous_row[-1]


def normalized_similarity(tokens_a: list, tokens_b: list) -> float:

    max_len = max(len(tokens_a), len(tokens_b), 1)

    if max_len == 0:
        return 1.0

    distance = levenshtein_distance(tokens_a, tokens_b)

    return 1.0 - (distance / max_len)


def longest_common_prefix_length(tokens_a: list, tokens_b: list) -> int:

    length = 0

    for token_a, token_b in zip(tokens_a, tokens_b):
        if token_a != token_b:
            break
        length += 1

    return length


def compute_avg_edit_distance(
    reference_decodings: list[list[str]],
    candidate_decodings: list[list[str]],
) -> float | None:

    if not candidate_decodings:
        return None

    total_distance = 0.0
    sample_count = 0

    for ref_texts, cand_texts in zip(reference_decodings, candidate_decodings):
        ref_texts = ref_texts or [""]
        cand_texts = cand_texts or [""]

        ref_best = ref_texts[0]
        cand_best = cand_texts[0]

        ref_best_tokens = ref_best.split()
        cand_best_tokens = cand_best.split()

        distance = levenshtein_distance(ref_best_tokens, cand_best_tokens)
        total_distance += distance
        sample_count += 1

    if sample_count == 0:
        return None

    return total_distance / sample_count


def summarize_similarity(
    reference_decodings: list[list[str]],
    candidate_decodings: list[list[str]],
) -> float | None:
    # pretty printing :)
    avg_distance = compute_avg_edit_distance(reference_decodings, candidate_decodings)
    if avg_distance is None:
        return None

    print("=" * 80)
    print("decoder similarity summary:")
    print(f"avg_edit_distance={avg_distance:.3f}")
    return avg_distance

