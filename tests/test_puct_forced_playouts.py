"""KataGo forced playouts + policy target pruning in `PuctBot` (docs/IDEAS.md R.10 M1).

The mechanism exists because the trained prior is a near point mass — mean top-1 0.817,
and only ~3.3 of the 8 kept root children carry a first-visit exploration bonus big
enough for PUCT to reach them at all — so the recorded visit counts largely re-encode
the prior and the search can never promote a move the prior ranked low. Forced playouts
guarantee every root child is evaluated; pruning takes that guarantee back out of the
target so the floor buys evaluation without flattening what gets distilled. The two
halves must move together, which is what these tests pin.

Both are pure functions of `(prior, visit counts)`, so they are exercised on stub nodes
rather than a loaded checkpoint — no policy/critic files, no search.
"""
import math
from types import SimpleNamespace

from src.services.bots.puct_bot import PuctBot


def _bot(k):
    """Just enough of a PuctBot for the three methods under test."""
    return SimpleNamespace(forced_playouts_k=k, c_puct=1.5, _root_node=None,
                           _forced_visits=lambda prior, total: PuctBot._forced_visits(
                               SimpleNamespace(forced_playouts_k=k), prior, total))


def _root(children, node_n=None):
    """A stub root: `children` maps action id -> (prior, visits)."""
    edges = {a: SimpleNamespace(prior=p, N=n) for a, (p, n) in children.items()}
    total = node_n if node_n is not None else sum(n for _, n in children.values())
    return SimpleNamespace(children=edges, N=total, mover=1, leaf_value=0.0)


def test_forced_visits_is_katago_sqrt_formula():
    me = SimpleNamespace(forced_playouts_k=2.0)
    # ceil(sqrt(2 * 0.5 * 200)) = ceil(sqrt(200)) = 15
    assert PuctBot._forced_visits(me, 0.5, 200) == 15
    # A low-prior child still gets a real floor, which is the whole point.
    assert PuctBot._forced_visits(me, 0.01, 300) == 3


def test_forced_visits_grows_sublinearly_in_the_budget():
    me = SimpleNamespace(forced_playouts_k=2.0)
    small = PuctBot._forced_visits(me, 0.2, 100)
    big = PuctBot._forced_visits(me, 0.2, 10000)
    assert big > small
    assert big < 10 * small  # 100x the budget buys 10x the floor, not 100x


def test_pruning_is_a_no_op_when_forced_playouts_are_off():
    root = _root({1: (0.8, 250), 2: (0.1, 30), 3: (0.1, 20)})
    counts = PuctBot._pruned_root_visits(SimpleNamespace(forced_playouts_k=0.0), root)
    assert counts == {1: 250, 2: 30, 3: 20}


def test_pruning_zeroes_a_child_that_only_ever_got_its_forced_quota():
    me = _bot(2.0)
    # total 300; child 3's quota is ceil(sqrt(2*0.1*300)) = 8, and it has exactly 8.
    root = _root({1: (0.8, 262), 2: (0.1, 30), 3: (0.1, 8)}, node_n=300)
    pruned = PuctBot._pruned_root_visits(me, root)
    assert pruned[3] == 0
    assert pruned[2] == 30 - 8  # kept only the visits PUCT spent voluntarily
    assert pruned[1] == 262  # the most-visited child is never pruned


def test_pruning_never_empties_the_target():
    me = _bot(2.0)
    # Every child sits exactly on its quota — a fully uninformative search.
    root = _root({1: (0.5, 18), 2: (0.3, 14), 3: (0.2, 12)}, node_n=320)
    pruned = PuctBot._pruned_root_visits(me, root)
    assert sum(pruned.values()) > 0
    assert pruned[1] == 18  # the best child survives whatever happens to the rest


def test_pruning_sharpens_rather_than_flattens():
    def entropy(counts):
        total = sum(counts.values())
        ps = [c / total for c in counts.values() if c > 0]
        return -sum(p * math.log(p) for p in ps)

    me = _bot(2.0)
    root = _root({1: (0.6, 180), 2: (0.2, 60), 3: (0.1, 10), 4: (0.1, 10)}, node_n=260)
    raw = {a: e.N for a, e in root.children.items()}
    pruned = PuctBot._pruned_root_visits(me, root)
    assert entropy(pruned) < entropy(raw)


def test_select_prefers_a_root_child_below_its_forced_quota():
    # Child 2 sits under its quota (5 visits against ceil(sqrt(2*0.1*300)) = 8) and looks
    # actively bad on the evidence so far (Q = -1), so plain PUCT picks child 1 — which is
    # exactly the case the quota has to override, and the reason a low-prior move never
    # gets evaluated without it.
    node = _root({1: (0.8, 250), 2: (0.1, 5), 3: (0.1, 45)}, node_n=300)
    node.children[1].W = 125.0  # Q = +0.5
    node.children[2].W = -5.0   # Q = -1.0
    node.children[3].W = 0.0
    forced = _bot(2.0)
    forced._root_node = node
    plain = _bot(0.0)
    plain._root_node = node
    assert PuctBot._select(plain, node, root_player=1) == 1
    assert PuctBot._select(forced, node, root_player=1) == 2


def test_select_ignores_quotas_away_from_the_root():
    # The invariant is "identical to plain PUCT", not any particular child: away from the
    # root a low-visit child can win on its own U term anyway, so comparing against the
    # k=0 bot is what actually distinguishes the quota from ordinary selection.
    node = _root({1: (0.9, 200), 2: (0.05, 1), 3: (0.05, 40)}, node_n=241)
    for edge in node.children.values():
        edge.W = 0.0
    forced = _bot(2.0)
    forced._root_node = _root({9: (1.0, 1)})  # the real root is some other node
    plain = _bot(0.0)
    plain._root_node = node
    assert PuctBot._select(forced, node, root_player=1) == PuctBot._select(plain, node, root_player=1)


def test_select_falls_back_to_puct_once_every_quota_is_met():
    node = _root({1: (0.9, 200), 2: (0.05, 30), 3: (0.05, 30)}, node_n=260)
    for edge in node.children.values():
        edge.W = 0.0
    forced = _bot(2.0)
    forced._root_node = node
    plain = _bot(0.0)
    plain._root_node = node
    assert PuctBot._select(forced, node, root_player=1) == PuctBot._select(plain, node, root_player=1)
