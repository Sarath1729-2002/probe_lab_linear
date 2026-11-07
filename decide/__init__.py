"""Decision run: three cheap, decisive experiments, each with a kill criterion fixed in advance.

A  Detection vs propagation: does the Goel et al. ~0.5 vs paper_v1 0.91-0.94 disagreement come from
   the readout (reusing a detection score vs training on the outcome) or from where the error sits
   (the model's own reasoning vs the user's message)?
B  Knowledge confound: do truth-probe failures concentrate on facts the model does not know?
C  Silent acceptance gate: given a false premise it knows is false, does the model sometimes adopt it
   and sometimes reject it (both classes needed before any probing project)?
"""
