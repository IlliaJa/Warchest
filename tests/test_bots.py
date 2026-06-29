"""Bots: RandomBot and GreedyBot must only ever return actions that are legal in
the current observation's mask, and the env must accept them (incl. P2's rotated
frame and any pending sub-turn they wander into).
"""
from src.services.environment.warchest_env import WarChestEnv
from src.services.bots.greedy_bot import GreedyBot
from src.services.bots.random_bot import RandomBot


def test_both_bots_return_legal_actions():
    env = WarChestEnv()
    obs, _ = env.reset()
    for bot in (RandomBot(), GreedyBot()):
        for _ in range(40):
            a, _, _ = bot.act(obs)
            assert obs['valid_action_mask'][a] == 1
            env_a = WarChestEnv.remap_action(a) if env.active_player == 2 else a
            obs, _, t, tr, info = env.step(env_a)
            assert info['action'].is_valid
            if t or tr:
                obs, _ = env.reset()
