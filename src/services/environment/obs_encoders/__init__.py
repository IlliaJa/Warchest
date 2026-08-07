"""Versioned observation encoders (docs/history.md — measurement infra).

Each encoder owns everything version-specific about turning the canonical game
state into a policy/critic observation (plane layout, normalizers, ego rotation,
feature derivation). The engine (WarChestEnv) stays obs-version-agnostic and
delegates to whichever encoder it was constructed with; agents in the round-robin
gauntlet each carry their own encoder and apply it to the shared game state.

To add a version: create `v<N>.py` with an `ObsEncoder<N>` class (exposing
`version`, `board_channels`, `global_dim`, `priv_dim`, `observation_space()`,
`encode(view)`, `encode_privileged(view)`), then register it below.
"""
from .v10 import ObsEncoderV10
from .v11 import ObsEncoderV11

# version int -> encoder class
ENCODERS = {
    10: ObsEncoderV10,
    11: ObsEncoderV11,
}

LATEST_VERSION = max(ENCODERS)


def get_encoder(version):
    """Return a fresh encoder instance for the given obs version."""
    try:
        return ENCODERS[version]()
    except KeyError:
        raise ValueError(
            f'unknown obs version {version}; registered: {sorted(ENCODERS)}'
        ) from None


def latest_encoder():
    """Return a fresh instance of the newest registered encoder."""
    return ENCODERS[LATEST_VERSION]()
