# Capacity group-prior control (not the final IMP model)

Copy of the controlled two-epoch A→W candidate-capacity pilot, restricted to
seed1 / capacity18 / fixed warm epoch4. Only known logits unchanged and unknown
logits shifted by log(2/18). Reference2 comes from the original control capacity,
not target-label bias tuning. Same data, losses, handoff and adaptation budget.

`CLS.capacity_log_prior` is a persisted buffer, not a trainable parameter. Warm
loading explicitly permits ONLY this new missing key. Final checkpoint replay
must use this model class and the saved buffer; ignoring it changes predictions.

This controls duplicated-slot GROUP mass, not all capacity-dependent effects.
Flat entropy, unknown pseudo-label CE, and maximum individual-slot decisions
still depend on capacity. A poor result is retained, not fixed by a tuned bias.
It is not DPMM inference or proof of semantic unknown class count.

Run all modified-model preflight/collection workers in fresh subprocesses and
assert module paths: notebook sys.modules may still contain older model classes.
Keep the previous raw-capacity pilot code/results untouched. Three-task full
baseline/IMP comparisons remain required after validating the modeling path.
