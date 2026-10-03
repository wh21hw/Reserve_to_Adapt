# Controlled capacity pilot (not a full baseline)

Office-31 A→W only, seed1, two adaptation epochs (indices4–5) from the fixed
epoch4 RTA warm checkpoint. Three predeclared arms: original unknown head2,
supported candidate head2, supported candidate head18. Candidate support floor5,
primary gate seed1, no target-label choice; capacity18 is NOT semantic class count.

Shared handoff reconstructs source soft bank/GMM/virtual templates from frozen
center-crop features, restores known/non-head SGD states, resets unknown-row
momentum in ALL arms, restores scheduler step56 and GRL step112, and resets RNG1.
This is controlled post-warmup adaptation, not uninterrupted official resume.
The discarded original untrained virtual pass is retained for minimal source
changes; all shared effective states are replaced by the handoff afterward.

All arms retain RTA adaptation losses and budget. The isolated optimizer manager
does not step on exceptions; nonfinite loss/gradients abort without auto retry.
Fixed final epoch6 is primary, target-label oracle-best is diagnostic only.
Full70-epoch matched comparisons and the Office-Home/VisDA tasks remain required.
