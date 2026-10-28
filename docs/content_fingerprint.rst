Binding caller-supplied frozen recipes
=====================================

``content_fingerprint(value, context=None).value`` returns the canonical SHA256
content identifier used by the library's data-only simulation recipes. R uses
``content_fingerprint(value, context=NULL)$value``. For example, use the result
on the complete supplied parameter object for a recipe's ``parameter_hash`` and
on the ScenarioCube value for its ``cube_hash``. A producer's existing digest
should not be recomputed over a different envelope. Exclude the digest field
itself when its contract binds the remaining payload.

The encoding is the existing ``creditratingsystem-canonical-v1`` contract:
object keys are nonempty printable ASCII and sorted; string values use UTF-8;
finite numbers use big-endian binary64 with negative zero normalized to zero.
Equivalent integer and floating values share a digest. Array order matters,
and booleans, numbers, strings, arrays, objects and null are distinct. Python
integers outside the exact binary64 range must be supplied as strings.

In R use named lists for objects, unnamed lists for arrays and NULL for null.
Length-one atomic vectors are scalars, so use ``list(100)`` for a one-element
numeric array. ``structure(list(), names=character())`` is an empty object;
``list()`` is an empty array. These distinctions must be preserved when loading
external JSON; ``jsonlite::fromJSON(..., simplifyVector=FALSE)`` is appropriate.

The digest identifies content. It neither authenticates an author nor checks
financial schema, provenance assertions or regulatory eligibility. The receiving
calculation separately validates the relevant contract. No files are written.
