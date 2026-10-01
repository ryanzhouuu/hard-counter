# Mechanics review: October 1, 2026

Current training attributes use `card-attributes:2026-10-01-r2`. This review corrects
seven coarse roles; it does not certify every card or establish model improvement.
Costs, role column order, card identities, and base-role inheritance are unchanged.

| Card | Correction | Evidence date / release date |
| --- | --- | --- |
| Furnace | Remove building; add air defense | [Supercell rework](https://supercell.com/en/games/clashroyale/blog/release-notes/new-season-fired-up/), August 4, 2025; [RoyaleAPI original gameplay preview](https://royaleapi.com/blog/furnace-rework-2025-august?lang=en) specifies ground troop and ground/air targeting. |
| Goblin Hut | Add air defense; retain building | [Supercell](https://supercell.com/en/games/clashroyale/blog/release-notes/new-season-fired-up/) confirms the reworked anti-air building as of August 4, 2025. |
| Zappies | Add air defense | [Supercell balance update](https://supercell.com/en/games/clashroyale/blog/release-notes/balance-update-coming-1-24/) adds ground and air targeting, January 24, 2018. |
| Mother Witch | Add air defense | [RoyaleAPI original release preview](https://royaleapi.com/blog/season18?lang=en) lists ground and air targets for the December 7, 2020 release. |
| Royal Delivery | Add air defense; retain spell | [Supercell balance notes](https://supercell.com/en/games/clashroyale/blog/release-notes/season-14-balance-changes/) describe ground/air area damage and a spawned troop, August 4, 2020. |
| Goblin Curse | Add air defense; retain spell | [Supercell balance notes](https://supercell.com/en/games/clashroyale/blog/release-notes/october-balance-changes/) explicitly describe damage to Bats, October 8, 2024. Air targeting is inferred from that interaction. |
| Rage | Add air defense; retain spell | [RoyaleAPI original update preview](https://royaleapi.com/blog/2022-q4-update?lang=en) explicitly confirms instant damage to air units for the December 12, 2022 rework. |

These labels mean a card has an air-response channel, not that it stops an air
push alone. Furnace's direct attack and spawned Fire Spirit are different damage
channels. Mother Witch's targeting does not make her a splash attacker. Ability
costs, activation, timing, and conditional deployments are not encoded by these
six role columns.

## Research mechanics

Research uses a separate versioned, field-level catalog. Each verified field
records its source URL, effective evidence date, unit, and normalization level
where applicable. Supercell announcements provide publisher evidence; the
linked RoyaleAPI release previews provide original gameplay/stat observations
for targeting omitted from the announcements. Older balance numbers from these
sources must not be treated as current level-16 damage or timing measurements.

The `official-static-partial:2026-10-01-r2` snapshot applies to September 24 through
October 1. The [September balance notes](https://supercell.com/en/games/clashroyale/blog/release-notes/september-balance-changes-2027/)
change damage/timing for reviewed cards, not the recorded structural predicates.
The snapshot adds the corrected targeting channels, explicitly excludes
Furnace from defensive buildings, records Royal Delivery's separate spell and
spawn capabilities, and records Minion Giant as airborne and building-targeting.
Hero Ice Wizard is marked conditional without treating his ability as an
unconditional answer. The September and first October catalogs remain byte-for-byte unchanged.

The second October revision also records Rage's air damage and
[Cannoneer's air/ground targeting](https://royaleapi.com/blog/cannoneer-january-2024?lang=en),
effective January 1, 2024. Freeze damage and control are sourced from the September
2026 balance notes; its targeting remains unknown. Minion Giant's building-only
targeting excludes air troops from its direct attack channel.

The [public RoyaleAPI bulk export](https://github.com/RoyaleAPI/cr-api-data/commit/d5461b0a59bff33c4da2fc845b07275b66b2d6ff)
was last updated October 18, 2023 when checked. It is not sufficient evidence for
current-era completeness. A current export or independently verified current
mechanics is needed to close the remaining field gaps.

Omitted fields remain unknown. Do not infer a false flag from an empty role set
or copy every base mechanic onto an evolution or hero. Hero abilities and evolved
deployment effects need explicit activation semantics before they can count as
ordinary defensive channels. Spirit Empress keeps conditional three/six cost
metadata and unknown unconditional airborne status. Tower deployment costs remain
inapplicable; missing tower damage and recharge measurements remain unknown.

Rage's control absence, Freeze targeting, spawned-unit damage channels, and unaudited
form/tower capabilities still need field-level verification. The partial catalog
is a preparation input, not evidence that a complete threat/response experiment
is ready. Freeze reviewed inputs and regenerate schemas/caches before any future
model comparison; retain historical catalog snapshots and experiment artifacts.
