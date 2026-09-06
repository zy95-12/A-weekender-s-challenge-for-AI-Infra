"""TEST FIXTURE ONLY: zero-weight service on explicit test ports.

Production `poc up` never imports this module and has no dummy-weight flag.
This fixture tests IPC/RPC, APIs and cancellation while weights download.
"""
import torch
from split_poc.runtime import PartialModel


def zero_weights(self, path):
    with torch.no_grad():
        for param in self.parameters():
            param.zero_()


PartialModel.load_owned_weights = zero_weights

if __name__ == "__main__":
    import sys
    if "--port" not in sys.argv or sys.argv[sys.argv.index("--port") + 1] not in {"18000", "18001"}:
        raise SystemExit("Test fixture requires port 18000 or 18001")
    print("TEST FIXTURE: zero weights, not real-model acceptance", flush=True)
    from split_poc.server import main
    main()
