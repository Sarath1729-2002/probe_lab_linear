from errprop.arith import (parse_equations, answer_equation, ancestors, candidate_injections,
                           select_injections, propagate, corrupt_value, format_like, numbers_in)
from errprop.labels import extract_answer, outcome, recovery_type, label_continuation, contamination_horizon

TRACE = (
    "Janet's ducks lay 16 eggs. She uses 3 + 4 = 7 eggs per day.\n"
    "Remaining eggs: 16 - 7 = 9.\n"
    "She also counts her ducks: 5 * 2 = 10 legs on the pond.\n"
    "Money: 9 * 2 = 18 dollars.\n"
    "#### 18"
)
QUESTION = "Janet's ducks lay 16 eggs per day. She eats 3 and bakes with 4. Sells at $2. 5 ducks?"


def test_parse_and_graph():
    eqs = parse_equations(TRACE)
    assert [e.rhs_value for e in eqs] == [7, 9, 10, 18]
    k = answer_equation(eqs, 18)
    assert k == 3
    assert ancestors(eqs, k) == {0, 1}          # the duck-legs line is a dead end
    assert eqs[1].parents == [None, 0]


def test_propagation_exact():
    eqs = parse_equations(TRACE)
    ch = propagate(eqs, 0, 6)                    # 3 + 4 = 6
    assert ch[1] == 10 and ch[3] == 20           # 16-6=10, 10*2=20


def test_candidates_on_and_off_path():
    eqs = parse_equations(TRACE)
    qn = numbers_in(QUESTION)
    cands = candidate_injections(eqs, 18, qn)
    on = [c for c in cands if c.on_path]
    off = [c for c in cands if not c.on_path]
    assert all(c.changes_answer for c in on)
    assert all(not c.changes_answer for c in off)
    assert any(c.eq_idx == 2 for c in off)       # dead-end equation is an off-path control
    # 9 + 1 = 10 collides with the dead-end '10 legs', so plus_one is filtered as ambiguous
    assert not [c for c in on if c.eq_idx == 1 and c.kind == "plus_one"]
    p1 = [c for c in on if c.eq_idx == 1 and c.kind == "times_ten"][0]
    assert p1.new_value == 90 and p1.propagated_answer == 180
    sel = select_injections(cands, max_on=3, max_off=1, seed=0)
    assert sum(c.on_path for c in sel) <= 3 and sum(not c.on_path for c in sel) <= 1


def test_corrupt_and_format():
    assert corrupt_value(48, "48", "digit_swap") == 84
    assert corrupt_value(44, "44", "digit_swap") is None
    assert corrupt_value(2.5, "2.50", "plus_one") == 2.51
    assert format_like(1300, "1,200") == "1,300"
    assert format_like(2.5, "2.50") == "2.50"


def test_latex_and_chains():
    t = r"First \(3 \times 4 = 12\). Then total = 12 + 8 = 20, and \frac{20}{4} = 5."
    eqs = parse_equations(t)
    assert [e.rhs_value for e in eqs] == [12, 20, 5]
    assert "1,200" not in t
    t2 = "Cost is 1,200 + 300 = 1,500 dollars."
    assert parse_equations(t2)[0].rhs_value == 1500


def test_wrong_arithmetic_is_dropped():
    assert parse_equations("so 3 + 4 = 8 apples") == []
    assert parse_equations("x = 5 and y = 7") == []


def test_answers_and_labels():
    assert extract_answer("blah\n#### 1,234") == 1234
    assert extract_answer(r"so \boxed{\frac{3}{4}}") == 0.75
    assert extract_answer("The answer is 42.") == 42
    assert outcome(20, 18, 20) == "FOLLOW"
    assert outcome(18, 18, 20) == "RECOVER"
    assert outcome(17, 18, 20) == "OTHER"
    assert outcome(18, 18, 18) == "NA_RECOVER"
    assert recovery_type(" eggs. Wait, 3 + 4 = 7.", 7) == "explicit"
    assert recovery_type(" eggs. 16 - 7 = 9 left", 7) == "restated"
    assert recovery_type(" eggs. 9 * 2 = 18", 7) == "silent"


def test_label_continuation():
    inj = {"propagated_answer": 20, "new_value": 6, "true_value": 7, "changed_values": [6, 10, 20]}
    cont = " eggs per day.\nRemaining: 16 - 6 = 10.\nMoney: 10 * 2 = 20.\n#### 20"
    r = label_continuation(cont, 18, inj)
    assert r["outcome"] == "FOLLOW" and r["uptake"] and r["horizon"] == 2
    cont2 = " eggs per day.\nRemaining: 16 - 7 = 9.\n9 * 2 = 18\n#### 18"
    r2 = label_continuation(cont2, 18, inj)
    assert r2["outcome"] == "RECOVER" and r2["recovery_type"] == "restated" and r2["horizon"] == 0
    assert contamination_horizon("10 * 2 = 20", [10]) == 1


def test_rounds_to_gold():
    from errprop.analysis import rounds_to_gold
    assert rounds_to_gold(59.9667, 60)        # followed + rounded = looks like recovery
    assert not rounds_to_gold(61, 60)
    assert rounds_to_gold(0.751, 0.75) and not rounds_to_gold(0.8, 0.75)
