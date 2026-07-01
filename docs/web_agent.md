# Web Agent: Playing warchestonline.com with the Trained Model

Goal: load a saved model checkpoint, play 5–15 games against the site's AI while
watching the full UI, with human-like pacing so the site is not stressed.

---

## 1. Tool choice — Playwright (non-headless), not raw HTTP

Even though the site is a REST API, **Playwright in non-headless mode is the right
choice here** for three reasons:

1. **You can watch** — it drives a real Chromium window; the game UI updates as the bot
   plays, exactly as if you were sitting there.
2. **Looks human** — real browser fingerprint, real mouse events, real user-agent.
   A bare `httpx` client from Python looks like a script; Playwright looks like a user.
3. **Login is free** — you log in manually in the browser window once, then the bot
   takes over. No cookie extraction needed.

The REST calls (`/units`, `/actions`, `/command`) happen via `page.evaluate(fetch(…))`
inside the Playwright session, so they carry the same cookies and session as the
browser — no auth wiring needed.

---

## 2. Respectful use checklist

**Check the ToS first.** Look for a "Terms of Service" or "Fair Play" page on
warchestonline.com and check whether automated play is prohibited. If it is, stop here.

If it's not prohibited (or silent):

| Practice | Why |
|----------|-----|
| Use your real account | Hiding who you are is more suspicious than using a bot |
| 1–3 s random delay between actions | Matches human reaction time; avoids rate-limit triggers |
| 30–60 s pause between games | Prevents rapid game cycling that looks like load testing |
| One game at a time | No concurrent sessions |
| Play during off-peak hours | Reduces any server load impact |
| Stop at 15 games | Self-imposed cap; more than enough to evaluate the model |

**VPN**: for 5–15 games on your own account, a VPN adds no meaningful protection and
may actually look more suspicious (datacenter IP). Skip it. The site knows who you are
from your account, not your IP. Only reconsider if you hit an IP-level block, which is
unlikely at this scale.

---

## Phase 0 — Recon findings (from DevTools Network tab)

### Endpoints confirmed

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/units?gameId=…` | own hand / bag / facedown coins |
| GET | `/actions?gameId=…` | legal actions + anti-replay hash |
| GET | `/log?gameId=…` | move history |
| POST | `/command` | submit a move |

### `/units` response shape

```json
{
  "unitsInHand":   [ { "id": 58602107, "type": "MARSHAL", "typeId": 10,
                       "quantity": 5, "state": "in_hand", "position": null } ],
  "unitsInBag":    [ … ],
  "facedownUnits": [ … ]
}
```

`position` is `null` for hand/bag coins; must be non-null for deployed units —
**format not yet seen** (open question #1).

### `/actions` response shape

```json
{
  "actions": [
    { "type": "pass",    "unitId": 58602107, "targetPositions": [] },
    { "type": "recruit", "unitId": 58602107, "targetPositions": [],
      "friendlyUnitIds": [58602104, …] }
  ],
  "hash": "ada6bd3b-…",
  "decidingPlayerId": 4640497,
  "lastActionTime":  "2026-06-29T17:58:54+00:00"
}
```

`hash` is a per-turn anti-replay token. `decidingPlayerId` drives turn detection.

### `/command` POST body (confirmed for `recruit`)

```json
{ "hash": "ada6bd3b-…", "actionType": "recruit", "gameId": 2531301,
  "playerId": 4640497, "unitId": 58602107, "friendlyUnitId": 58602119,
  "recruitType": "faceup", "targetPosition": null, … }
```

### Site type → internal DECK (partial)

| Site `type` | `typeId` | Internal |
|-------------|---------|----------|
| `MARSHAL` | 10 | Marshal |
| `SWORDSMAN` | 17 | Swordsman |
| `WARRIOR_PRIEST` | 18 | Warrior Priest |
| `ROYAL_COIN_WHITE` | 13 | Royal (white) |

---

## Open questions (still needed from DevTools)

1. **Deployed unit `position` format** — open `/units` when units are on the board.
   What does `"position"` look like for a deployed unit? Probably `{"col": 3, "row": 2}`.

2. **`targetPositions` for move/attack** — make a move and capture `/actions`. This
   array will reveal the hex coordinate format used in spatial actions.

3. **Board terrain endpoint** — `/units` has no terrain or base data. Check what other
   requests fire on page load. Look for `/board`, `/game/{id}`, or `/state`.

4. **`/command` body for a move** — make a move manually, capture the POST body.
   Confirm `targetPosition` format and field names for spatial actions.

---

## Architecture

```
┌───────────────────────────────────────────────┐
│   Playwright (non-headless Chromium window)   │
│   — user watches the game here —              │
└──────┬────────────────────────────────────────┘
       │ page.evaluate(fetch(…))  ← carries session cookies automatically
┌──────▼───────────────┐
│   WarchestApiClient  │   thin wrapper around page.evaluate fetch calls
│   get_units()        │
│   get_actions()      │
│   post_command()     │
└──────┬───────────────┘
       │ raw JSON
┌──────▼────────────┐
│  StateObserver    │   JSON → obs tensor (board + globals + mask)
│  coin_map.py      │   site typeId ↔ internal UNIT_IDS index
└──────┬────────────┘
       │ obs, mask
┌──────▼────────────┐
│   PolicyModel     │   loaded .pt checkpoint
│   → action_id     │
└──────┬────────────┘
       │ flat action id (0–1874)
┌──────▼────────────┐
│  ActionEncoder    │   action id → /command POST body
└───────────────────┘
```

---

## Implementation phases

### Phase 1 — `coin_map.py`
Full mapping of site `typeId` / `type` string → internal `DECK` index and `UNIT_IDS`
index. Needs a complete `/units` response (start a game, copy the full JSON).

### Phase 2 — `StateObserver`
File: `src/services/web_agent/observer.py`

Calls `get_units()` + board endpoint (TBD) + `get_actions()`, builds
`(obs_tensor, legal_mask)` matching OBS_VERSION 4. Blocked on open questions #1–3.

### Phase 3 — `ActionEncoder`
File: `src/services/web_agent/actor.py`

Decodes flat action id → verb + hex coords → `/command` body.

| Internal verb | Site `actionType` | Key POST fields |
|---|---|---|
| move (0–5) | `"move"` | `unitId`, `targetPosition` |
| attack (6–11) | `"attack"` | `unitId`, `targetPosition` |
| deploy (14–29) | `"deploy"` | `unitId` (hand coin), `targetPosition` |
| pass | `"pass"` | `unitId` |
| recruit | `"recruit"` | `unitId`, `friendlyUnitId`, `recruitType` |
| claim_initiative | `"claimInitiative"` | `unitId` |

### Phase 4 — main loop
File: `src/services/web_agent/agent.py`

```python
async def run(page, policy, player_id, game_id, max_games=10):
    for _ in range(max_games):
        await play_one_game(page, policy, player_id, game_id)
        await asyncio.sleep(random.uniform(30, 60))  # pause between games

async def play_one_game(page, policy, player_id, game_id):
    while True:
        resp = await client.get_actions(page, game_id)
        if resp["decidingPlayerId"] != player_id:
            await asyncio.sleep(3)
            continue
        if game_over(resp):
            break
        obs, mask = await observer.build(page, game_id, player_id)
        action_id = policy.select_action(obs, mask)
        body = encoder.encode(action_id, resp, ...)
        await asyncio.sleep(random.uniform(1.0, 2.5))  # human-like delay
        await client.post_command(page, body)
```

### Phase 5 — entry point
File: `src/app/web_play.py`

```bash
python src/app/web_play.py --checkpoint runs/best.pt --games 10
```

The script opens Chromium visibly, navigates to the site, pauses for you to log in
manually, then starts the loop.

---

## File layout

```
src/
  services/
    web_agent/
      __init__.py
      client.py      # page.evaluate fetch wrappers
      observer.py    # JSON → obs tensor
      actor.py       # action id → POST body
      coin_map.py    # site typeId ↔ internal UNIT_IDS / DECK index
  app/
    web_play.py      # entry point
```

## Dependencies to add

```
playwright   # pip install playwright && playwright install chromium
```
