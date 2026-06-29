# Unit reference

Transcription of all 16 War Chest unit cards, for implementing the full roster
(Phase 3 = vanilla unit variety, Phase 4 = tactics & attributes — see
`docs/full_game_plan.md`). Rules text is reproduced as a development reference only;
**do not embed the original coin artwork** (it's AEG / Brigette Indelicato's IP — see the
visualisation discussion). The "glyph" column is an *original look-alike* symbol for the
renderer, not their art.

## Conventions

- **Coins (`xN`)** = total coins of that unit a player owns. At setup **2 go in the bag**,
  the **rest go to supply** (recruitable). So `INITIAL_BAG = 2` per unit type and
  `SUPPLY = N - 2` (x4 → supply 2, x5 → supply 3). The Royal coin is `x1`, bag only.
- **Category:** `TACTIC` = an active maneuver paid by discarding the coin face-up;
  `ATTRIBUTE` = a passive/triggered ability (no extra cost); `RESTRICTION` = a limit on
  the unit. A unit can have more than one.
- **Maneuver** = move / attack / control / use-tactic (the face-up discard actions).
- **Glyph reliability:** ✅ monochrome BMP (renders in matplotlib's default font);
  ⚠ color emoji (needs an emoji font — may show as a box; use the letter fallback). The
  implementer may instead draw simple original icons.
- **Currently implemented (2026-06-28):** **all 16 units have their tactics, attributes, and
  restrictions** (Phase 4 complete — see `docs/full_game_plan.md`). Coin ids run 1..16 for the
  units (in the table order below: Swordsman=1, Knight=2, … Warrior Priest=16) and 17 for the
  Royal coin. Mechanics are named generically in `roster.py` (`tactic` + `tactic_params` and the
  boolean attribute flags) so they are reused across units and future DLC; the renderer still
  uses the short letter glyphs (`Sw`, `Kn`, …) rather than the card art.

---

## Roster overview

| Unit | Coins | Bag/Supply | Glyph | Coin color (approx) | Coin depicts | Category |
|---|---|---|---|---|---|---|
| Swordsman | x5 | 2 / 3 | ⚔ ✅ | navy blue `#2b3f6b` | crossed swords | ATTRIBUTE |
| Knight | x4 | 2 / 2 | 🛡 ⚠ (`Kn`) | sky blue `#3fa6dc` | armored figure w/ shields | ATTRIBUTE |
| Cavalry | x4 | 2 / 2 | ♞ ✅ | orange `#c87a2c` | rearing horse | TACTIC |
| Light Cavalry | x5 | 2 / 3 | 🐎 ⚠ (`Lc`) | yellow-green `#8fae3e` | running horse | TACTIC |
| Lancer | x4 | 2 / 2 | 🏇 ⚠ (`La`) | red `#c0392b` | mounted lancer | TACTIC + RESTRICTION |
| Archer | x4 | 2 / 2 | 🏹 ⚠ (`Ar`) | grey-teal `#5a9ea0` | bow & arrow | TACTIC + RESTRICTION |
| Crossbowman | x5 | 2 / 3 | ➹ ✅ | plum `#7d4f5e` | crossbow bolt | TACTIC |
| Berserker | x5 | 2 / 3 | 🪓 ⚠ (`Be`) | forest green `#2f5d3b` | crossed axes | ATTRIBUTE |
| Footman | x5 | 2 / 3 | ♟ ✅ | teal `#2c8090` | foot soldier w/ shield | TACTIC + ATTRIBUTE |
| Pikeman | x4 | 2 / 2 | 🔱 ⚠ (`Pk`) | gold `#d4a72c` | pike/spear | ATTRIBUTE |
| Ensign | x5 | 2 / 3 | ⚑ ✅ | olive `#9aa83c` | banner / flag | TACTIC |
| Marshall | x5 | 2 / 3 | ⚜ ✅ (`★`) | terracotta `#bf5a2f` | commander w/ standard | TACTIC |
| Mercenary | x5 | 2 / 3 | 🍺 ⚠ (`Me`) | maroon `#8c2f2f` | flagon (sellsword) | ATTRIBUTE |
| Scout | x5 | 2 / 3 | 🐦 ⚠ (`Sc`) | blue `#3a6ea8` | bird (raven/falcon) | ATTRIBUTE |
| Royal Guard | x5 | 2 / 3 | ♜ ✅ | rose `#d98c9c` | tower / keep | TACTIC + ATTRIBUTE |
| Warrior Priest | x4 | 2 / 2 | ✝ ✅ | purple `#6e4a8c` | Celtic cross | ATTRIBUTE |

Glyph notes: prefer the ✅ monochrome picks for reliable matplotlib rendering. Reassign the
two placeholders — **Swordsman ♖ → ⚔**, **Knight ♞ → 🛡/`Kn`** — which frees **♞ for Cavalry**
and **♜ for Royal Guard**.

Color notes: the hex values are **approximations read from the card art**, not official
values — they're a per-unit-type palette for tinting *original* icon discs in the renderer
(e.g. fill the disc with the coin color, keep the player identity on the border). Colours are
not themselves copyrightable, but don't lift the actual coin images. Note this is orthogonal
to the current renderer, which colors units by **player** (`P1` darkred / `P2` midnightblue);
coin color identifies the unit *type*, so use it on a separate visual channel.

---

## Cards (exact rules text + implementation notes)

### Vanilla-ish / movement tactics

**Swordsman** (x5) — ⚔
- ATTRIBUTE: *After the Swordsman attacks, it may move.*
- Impl: after a successful attack action by a Swordsman, grant an optional free move (a
  second maneuver with no coin cost) to the same unit. Triggers also when the attack is
  granted by Marshall (per FAQ).

**Knight** (x4) — 🛡
- ATTRIBUTE: *The Knight can only be attacked by units that are bolstered.*
- Impl: in attack legality, an enemy Knight is a valid target only if the attacker's stack
  height > 1 (bolstered). Affects `get_possible_actions` (mask) and `perform_attack`.

**Cavalry** (x4) — ♞
- TACTIC: *Move and then attack.* (one move into an adjacent cell, then an attack on an
  adjacent enemy — one coin pays for both)
- Impl: a `tactic` action = move 1 + attack; needs a tactic verb/sub-action.

**Light Cavalry** (x5) — 🐎
- TACTIC: *Move two spaces.* (may also use a normal 1-space move — FAQ)
- Impl: tactic = move up to 2 cells (path through empty cells). Normal move still available.

**Lancer** (x4) — 🏇
- TACTIC: *Move one or two spaces and then attack, all in a straight line.*
- RESTRICTION: *The Lancer can only attack by using its tactic.* (must both move and attack;
  cannot attack adjacent without moving — FAQ)
- Impl: tactic = straight-line move (1–2) then attack the unit at the end of the line; no
  normal attack action for the Lancer.

### Ranged tactics + restrictions

**Archer** (x4) — 🏹
- TACTIC: *Attack a unit two spaces away. The intervening space may be occupied by a unit.*
- RESTRICTION: *The Archer can only attack by using its tactic* (no adjacent normal attack).
- Impl: ranged attack at distance exactly 2 (any direction reachable in 2 hex steps);
  intervening occupancy ignored. No normal attack.

**Crossbowman** (x5) — ➹
- TACTIC: *Attack a unit two spaces away in a straight line. The intervening space cannot be
  occupied by a unit.* (May also make a normal adjacent attack — FAQ; no restriction.)
- Impl: ranged attack at distance 2 along a straight hex line with the middle cell empty;
  PLUS the normal adjacent attack remains legal.

### Repeating / chaining

**Berserker** (x5) — 🪓
- ATTRIBUTE: *After the Berserker maneuvers, you may maneuver it again by discarding a
  bolstered coin from the Berserker unit. You may do this multiple times, but you may not
  remove the final coin.*
- Impl: after any Berserker maneuver, optionally pay by removing one coin from its own stack
  (to the box) to take another maneuver; repeatable while stack height ≥ 2. The extra
  maneuvers are paid by stack coins, not hand coins (FAQ: only one hand coin total).

### Force-multipliers (grant an action to another unit)

**Ensign** (x5) — ⚑
- TACTIC: *Choose a friendly unit within two spaces of the Ensign. The chosen unit performs
  a normal move to a space within two spaces of the Ensign.* (a normal 1-space move — FAQ;
  Berserker's attribute can trigger off it — FAQ)
- Impl: tactic targets a friendly unit ≤2 away → grants it a 1-space move whose destination
  is also ≤2 from the Ensign.

**Marshall** (x5) — ⚜
- TACTIC: *Choose a friendly unit that is within two spaces of the Marshall. The chosen unit
  attacks, if able.* (grants a NORMAL attack only — cannot enable Archer/Lancer special
  attacks; on-attack attributes of the chosen unit do trigger — FAQ)
- Impl: tactic targets a friendly unit ≤2 away → that unit makes a normal adjacent attack.

### Special deploy / recruit

**Footman** (x5) — ♟
- TACTIC: *Perform one maneuver with each Footman unit on the board.* (the two maneuvers may
  differ — FAQ)
- ATTRIBUTE: *Two Footman units may be deployed at a time.* (the only unit allowed two copies
  on the board)
- Impl: relax "one unit per type" for Footman (allow 2); tactic does one maneuver per Footman
  on board.

**Scout** (x5) — 🐦
- ATTRIBUTE: *The Scout may be deployed adjacent to any friendly unit.* (not just onto
  controlled locations)
- Impl: extend deploy legality for the Scout to any empty cell adjacent to a friendly unit.

**Mercenary** (x5) — 🍺
- ATTRIBUTE: *After you recruit a Mercenary, you may maneuver your Mercenary unit.* (only if
  the Mercenary is already on the board; it's a free maneuver, not a deploy/recruit — FAQ)
- Impl: when a recruit action takes a Mercenary coin and a Mercenary is on the board, grant a
  free maneuver to it.

### Defensive / triggered attributes

**Pikeman** (x4) — 🔱
- ATTRIBUTE: *When the Pikeman is attacked by an adjacent unit, remove a coin from that unit.*
  (happens simultaneously with the attack, regardless of outcome; it is NOT itself an attack,
  so it can damage an attacking Knight — FAQ)
- Impl: in `perform_attack`, if the target is a Pikeman and the attacker is adjacent, also
  remove one coin from the attacker's stack (to the box).

**Royal Guard** (x5) — ♜
- TACTIC: *Discard the Royal Coin to move the Royal Guard up to 2 spaces to a location that
  you control.*
- ATTRIBUTE: *When the Royal Guard is attacked, you may remove a Royal Guard coin from the
  supply rather than from its unit.* (absorbs the hit from supply)
- Impl: a tactic paid specifically by the Royal coin (special pay-coin); defensive option to
  decrement supply instead of the on-board stack when attacked.

**Warrior Priest** (x4) — ✝
- ATTRIBUTE: *After the Warrior Priest attacks or controls, draw one coin from your bag and
  immediately use it to take any action.* (the drawn coin must be used immediately for one of
  the three action types; you may effectively "pass" it by discarding — FAQ. Triggers also
  when its attack is granted by Marshall — FAQ.)
- Impl: after a Warrior Priest attack/control, draw 1 from the bag and force an immediate
  bonus action with it.

---

## Cross-cutting mechanics these introduce

Implementing the roster mainly means adding these reusable mechanics (then each unit is a
small config on top):

1. **`tactic` action type** — a per-unit special maneuver, paid by the matching coin. Needs
   its own verb/sub-action and per-unit legality + effect (Cavalry, Light Cavalry, Lancer,
   Archer, Crossbowman, Ensign, Marshall, Royal Guard).
2. **Ranged attack** with distance-2 targeting and line/occupancy rules (Archer, Crossbowman).
3. **Multi-step / straight-line movement** (Light Cavalry, Lancer, Royal Guard, Ensign-grant).
4. **Grant-action-to-another-unit** (Ensign → move, Marshall → attack).
5. **Free / chained maneuvers** (Swordsman post-attack move, Berserker stack-paid repeats,
   Mercenary post-recruit maneuver, Warrior Priest bonus action).
6. **Attack-targeting modifiers** (Knight only attackable when bolstered).
7. **On-defense triggers** (Pikeman counter-coin, Royal Guard absorb-from-supply).
8. **Special deploy rules** (Scout adjacency, Footman two-copies).
9. **Restrictions** (Archer/Lancer: no normal attack — only via tactic).

These map onto the Phase 4 sub-step clusters in `docs/full_game_plan.md`; add a few units per
step, each with a focused test drawn from the FAQ edge cases (rulebook pp. 16–18).
