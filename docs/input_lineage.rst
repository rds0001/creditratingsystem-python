Local input, historical releases and ledger reconciliation
=========================================================

The ``examples/input_lineage.py`` and native ``examples/input_lineage.R``
workflows accept local account and release CSV paths plus explicit schemas,
control totals and knowledge cutoffs. They use ``read_local_dataset``,
``join_as_of``, ``content_fingerprint`` and ``validate_source_lineage``. The
result includes the prepared rows and the complete audit inputs for inspection
before passing financial features to a product assessment.

The supplied synthetic account control ledger has 31 accounts with balances
100, 200, ..., 3100 EUR: 49,600 EUR and 31 units of weight. A January observation
has an original January release and a February correction. The January view
uses the original release; a later knowledge view can use the correction while
keeping the observation date fixed. March observations remain ineligible. The
original view is then independently re-executed and must remain identical.
These amounts demonstrate source reconciliation; they are not product pricing
parameters or evidence that each complete product journey has been accepted.

The immutable archive hashes cover the input files, including later revisions.
Predictive lineage instead binds the selected historical source rows. This
separation prevents later releases from becoming hidden ancestors of a past
prediction. The right side of the audited join is the selected snapshot, not an
equality join to every release in the archive. Unmatched releases remain in the
archive and are not silently used as features. Missing coverage raises a visible
error; ambiguous releases also fail rather than using input order as a tie-break.

Bank users supply schemas with explicit types and numeric units, independently
controlled record identities and balances, and their dated release history.
``prepare_snapshot`` (R: ``prepare_input_snapshot``) returns the actual joined
rows, content hashes and audit. The provided reference control convention uses
``product_id`` plus a revision suffix to identify release rows; adapt this example
adapter to the bank's real ledger keys. The underlying APIs accept independent
explicit record identifiers. The ``verify_reference=False`` option disables
synthetic feature-value assertions, but ledger validation still runs and reports
failures. A caller must inspect its decision before using the rows for scoring.
Content hashes identify data; they do not authenticate its author or independently
prove that a source's declared availability date is true.
