# Model input contracts

The attention model uses `attention-card-schema:v1`. Catalog and attribute
versions describe the selected data; vocabulary sizes are derived from that
catalog rather than fixed to the June 2026 population. The packaged June catalog
and attributes remain the defaults.

## Catalog snapshots

Catalog JSON contains `catalog_version` and `entries`. Each entry is
`[source_id, source_name, card_id, form]`. Source IDs must be ordered from zero;
names and card/form identities must be unique. Evolution and hero entries need
a corresponding base or champion entry. Supported forms remain `base`,
`champion`, `evolution`, and `hero`.

Source IDs are lookup positions in this mapping, not model tokens. Model tokens
index the artifact's sorted card/form vocabulary. Adding identities may change
token numbers, so each artifact keeps its exact catalog, attributes, vocabulary,
feature definitions, and weights together.

## Schema validation

The schema checks catalog hashes and versions, complete base-card attributes,
vocabulary alignment, token indices, and attribute vectors. It supports up to
65,536 card/form identities. Roles, elixir scaling, and form inheritance retain
their existing meanings. Models still estimate deck matchups under an equal-skill
assumption; additional vocabulary entries alone do not establish trained support.

Training eligibility remains level 16. The current Kaggle prepared-data contract
also requires its ranked mode and winner-first rows. Supporting another catalog
does not change those source requirements.
