"""Small deterministic policies, independently testable without GPUs."""


def choose(active, policy, decode_rounds, quota):
    prefills=[j for j in active if not j.prefilled]
    decodes=[j for j in active if j.prefilled]
    if policy=="decode-first" and decodes and (not prefills or decode_rounds<quota):
        return decodes,"decode",decode_rounds+1
    if prefills:
        return prefills[:1],"prefill",0
    return decodes,"decode",decode_rounds+1
