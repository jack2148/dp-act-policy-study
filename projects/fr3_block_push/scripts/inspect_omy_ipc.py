#!/usr/bin/env python3
"""Inspect the latest valid OMY IPC state without blocking."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


STUDY_ROOT = Path(__file__).resolve().parents[3]
if str(STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT))

from projects.fr3_block_push.ipc.omy_state_client import OmyStateClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--stale-timeout", type=float, default=0.2)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--require-message",
        action="store_true",
        help="exit nonzero unless at least one valid OMY state is received",
    )
    args = parser.parse_args()
    client = OmyStateClient(
        args.endpoint,
        stale_timeout=args.stale_timeout,
    )
    deadline = time.monotonic() + args.duration
    last_sequence = None
    try:
        while time.monotonic() < deadline:
            state = client.read()
            if state.message is not None and state.message.sequence != last_sequence:
                message = state.message
                print(
                    f"sequence={message.sequence} fresh={state.fresh} "
                    f"age={state.age_seconds:.4f}s "
                    f"names={message.joint_names} position={message.position} "
                    f"trigger={message.trigger_position:.4f}",
                    flush=True,
                )
                last_sequence = message.sequence
            time.sleep(0.01)
    finally:
        client.close()
    print(f"rejected_messages={client.rejected_messages}")
    if args.require_message and last_sequence is None:
        print(
            f"no valid OMY state received from {args.endpoint} "
            f"within {args.duration:.1f}s",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
