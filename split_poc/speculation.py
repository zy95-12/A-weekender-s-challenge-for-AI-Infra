"""CPU prompt-lookup draft and greedy target verification.

No additional model, weights, GPU, or target fallback. A longest matching suffix
in the known prompt/output proposes its historical continuation. Acceptance is
always decided by the split target, including correction/bonus output.
"""

def propose(prefix, limit, max_ngram=8):
    if limit <= 0:
        return []
    for n in range(min(max_ngram,len(prefix)-1),1,-1):
        suffix=prefix[-n:]
        # Prefer the latest complete earlier occurrence. Never invent tokens.
        for start in range(len(prefix)-n-1,-1,-1):
            if prefix[start:start+n]==suffix:
                return prefix[start+n:min(start+n+limit,len(prefix))]
    return []


def confirm(draft, target, remaining, eos, ignore_eos):
    if len(target)!=len(draft)+1 or remaining<1:
        raise ValueError('Invalid greedy verification lengths')
    accepted=0
    while accepted<len(draft) and draft[accepted]==target[accepted]:
        accepted+=1
    output=(draft[:accepted]+[target[accepted]])[:remaining]
    if not ignore_eos:
        for i,token in enumerate(output):
            if token in eos:
                output=output[:i+1]
                break
    return output,min(accepted,len(output))
