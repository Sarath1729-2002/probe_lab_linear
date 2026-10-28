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


def test_new_operators_and_synth():
    import random
    from errprop.arith import corrupt_value, ALL_OPERATORS, candidate_injections, parse_equations
    from errprop.synth import serial_chains, bbh_arithmetic
    r = random.Random(0)
    v = corrupt_value(480, "480", "random_same_digits", rng=r)
    assert 100 <= v <= 999 and v != 480
    v = corrupt_value(480, "480", "random_scale", rng=r)
    assert v != 480 and 96 <= v <= 2400
    assert corrupt_value(15, "15", "copy_operand", operands=[3, 5]) == 3
    eqs = parse_equations("4 * -2 = -8\n-8 - 5 = -13")
    assert eqs[1].parents[0] == 0                     # signed operand links to its producer
    cands = candidate_injections(parse_equations(TRACE), 18, [16, 3, 4, 2, 5], kinds=ALL_OPERATORS, seed=1)
    assert {c.kind for c in cands} >= {"random_scale", "copy_operand"}
    ch = serial_chains(12, 0)
    assert {c["level"] for c in ch} == {3, 4, 5, 6, 7, 8} and all(c["gold"] > 0 for c in ch)
    assert all(b["question"].startswith("Compute") for b in bbh_arithmetic(5, 0))


def test_strict_number_rejects_symbolic_golds():
    from errprop.labels import strict_number
    assert strict_number("\\frac{3}{4}") == 0.75 and strict_number("1,234") == 1234
    assert strict_number("12\\text{ cm}") == 12 and strict_number("\\$15.50") == 15.5
    for bad in ("\\sqrt{2}", "3\\pi", "x^2+1", "(1,2)"):
        assert strict_number(bad) is None, bad


def test_mention_injection_and_tolerance():
    from errprop.arith import mention_injections
    from errprop.labels import label_continuation
    body = "The Battle of Hastings was fought in 1066 in England.\n"
    inj = mention_injections(body, 1066.0, [], ("plus_one", "times_ten"), seed=1)
    assert {i.kind for i, _, _ in inj} == {"plus_one", "times_ten"}
    i, a, b = inj[0]
    assert body[a:b] == "1066" and i.propagated_answer == i.new_value
    d = {**i.__dict__, "tol": 1e-4}
    assert label_continuation(f" so.\n#### {i.new_text}", 1066.0, d)["outcome"] == "FOLLOW"
    assert label_continuation(" no, 1066.\n#### 1066", 1066.0, d)["outcome"] == "RECOVER"
    assert mention_injections("In 1066 ...", 1066.0, [1066.0]) == []      # value given in the question
    # 2% tolerance (MMLU-Pro): 3.19 counts as gold 3.2
    d2 = {"propagated_answer": 6.4, "new_value": 4.0, "changed_values": [6.4], "true_value": 2.0, "tol": 0.02}
    assert label_continuation("#### 3.19", 3.2, d2)["outcome"] == "RECOVER"
