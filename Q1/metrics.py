"""Position-aligned exact recovery metrics; no bag-of-words matching."""
from collections import Counter


def score(target, prediction, tokenizer=None, seen_token_ids=None):
    assert len(target) == len(prediction)
    correct = [int(a == b) for a, b in zip(target, prediction)]
    counts = Counter(target)
    per_type = {t: sum(c for x, c in zip(target, correct) if x == t) / n for t, n in counts.items()}
    result = {'positions': len(target), 'correct': sum(correct), 'token_accuracy': sum(correct)/len(target),
              'distinct_token_types': len(counts), 'macro_token_type_accuracy': sum(per_type.values())/len(per_type),
              'exact_sequence_match': target == prediction}
    if tokenizer:
        special = set(tokenizer.all_special_ids)
        keep = [i for i,t in enumerate(target) if t not in special]
        result['non_special_token_accuracy'] = sum(correct[i] for i in keep)/len(keep) if keep else None
        result['exact_decoded_text_match'] = tokenizer.decode(target) == tokenizer.decode(prediction)
    if seen_token_ids is not None:
        keep = [i for i,t in enumerate(target) if t in seen_token_ids]
        result['training_seen_positions'] = len(keep)
        result['training_seen_token_accuracy'] = sum(correct[i] for i in keep)/len(keep) if keep else None
    return result
