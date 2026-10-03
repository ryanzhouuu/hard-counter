# Mechanics review: October 2, 2026

Current training attributes use `card-attributes:2026-10-01-r3`. This review corrects
eight coarse roles; it does not certify every card or establish model improvement.
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
| Freeze | Add air defense; retain spell | [User-supplied targeting and description](../experiments/mechanics/evidence/3951013bf78e8bb50c5b020f72c7d0b5972d6351c2fcd2fb186571b973e57031.json), October 1, 2026, confirms ground/air damage and freezing. |

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

Manual confirmations also count as known mechanics. They record the supplied
facts, report date, and a hash of the evidence record without requiring web URLs.
Descriptions are retained verbatim; they do not establish numeric combat stats.

The `mechanics-partial:2026-10-02-r7` snapshot applies to September 24 through
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
2026 balance notes; its targeting is now manually confirmed. Minion Giant's building-only
targeting excludes air troops from its direct attack channel.

The third revision records manually confirmed ground/air targeting for Tower
Princess, Dagger Duchess, Royal Chef, and Freeze. Cannoneer's targeting retains
its published source and is also corroborated in the manual record. Confirmed
spells have no primary airborne actor; summoned and ability-only threats remain
separate. All four towers and Freeze pass the ordinary air-targeting audit.

The [public RoyaleAPI bulk export](https://github.com/RoyaleAPI/cr-api-data/commit/d5461b0a59bff33c4da2fc845b07275b66b2d6ff)
was last updated October 18, 2023 when checked. It is not sufficient evidence for
current-era completeness. A current export, manual confirmations, or published
mechanics can close the remaining field gaps.

Omitted fields remain unknown. Do not infer a false flag from an empty role set
or copy every base mechanic onto an evolution or hero. Paid Hero abilities and
spawned/death stages need separate activation semantics. Explicitly sourced
automatic effects on evolved deployment can establish an ordinary channel.
Spirit Empress keeps conditional three/six cost
metadata and unknown unconditional airborne status. Tower deployment costs remain
inapplicable; missing tower damage and recharge measurements remain unknown.

The fourth revision completes ordinary air predicates for 185 of 186 identities.
Primary evidence is retained where available; remaining flags use the
[Clash Royale Wiki](https://clashroyale.fandom.com/wiki/Cards)'s explicit targeting,
transport, type, and effect descriptions. These are community observations,
not publisher confirmations. Their field dates record review on October 1, 2026,
not a claimed balance-change date. Earlier snapshots and existing verified/manual
facts remain unchanged. No quantitative combat measurements are inferred.

`targets_air` includes initially deployed attack components and intrinsic direct
effects: backpack Spear Goblins, Ram Rider's rider, Goblinstein's Doctor,
Goblin Machine's rocket, and Electro Giant's reflection. Automatic evolved effects
also count when explicitly sourced: [Cannon's deployment barrage](https://royaleapi.com/blog/cannon-evolution-new-card-2024-november?lang=en)
hits air despite its ground-only regular attack. Summoned children, death effects,
and paid Hero abilities remain separate. A channel indicates possible response,
not guaranteed defensive success or availability on every cycle.

`airborne` describes the primary ordinary deployment, including
[Evolved Royal Hogs' initial flight](https://royaleapi.com/blog/royal-hogs-evolution-2025-november?lang=en).
It does not promise flight for the unit's entire lifetime. [Hero Wizard](https://royaleapi.com/blog/hero-wizard-january-2026?lang=en)
is ground in this channel; his paid flight ability remains separate. Buildings are
verified non-airborne; spells have no primary actor and are inapplicable.

`spell_control` means restricting enemy movement or actions, including slow,
stun, freeze, displacement, or forced retargeting. Friendly buffs, damage
amplification, and death conversion do not qualify. Rage and Goblin Curse therefore
have damage without control. [Royal Delivery's removed knockback](https://supercell.com/en/games/clashroyale/blog/release-notes/season-14-balance-changes/)
and [Poison's restored movement slowdown](https://supercell.com/en/games/clashroyale/blog/release-notes/summer-update-balance-changes-1-2/)
are explicitly sourced.

[Spirit Empress](https://royaleapi.com/blog/spirit-empress-new-card-2025-july?lang=en)
is ground-only at three Elixir and flies/targets air at six. The new evidence
categories `airborne_condition` and `targets_air_condition` both record
`available_elixir_at_least_6`; her unconditional booleans remain unknown because
deck identities do not identify Elixir at deployment. The full-population strict
air gate still rejects this identity. Every other identity passes the ordinary
air gate; deployment-cost and quantitative tower gates remain separate.

The fifth revision records Spirit Empress as `airborne_mode: hybrid`. This optional
category accepts `ground`, `air`, or `hybrid`; hybrid describes her alternative
ground/air deployments. It is a verified classification derived from the sourced
form descriptions, not an assertion of the form deployed in any particular battle.
The fourth revision remains frozen, and the hybrid label does not override the
unconditional flags or satisfy the strict static air gate.

The sixth revision adds the following conditional channels, reviewed October 2.
Earlier snapshots and all their field values remain frozen. The supported
September 24–October 1 population window is unchanged.

| Identity | Conditional capability | Trigger | Response scope | Ability cost |
| --- | --- | --- | --- | --- |
| Spirit Empress | Flight and air damage | Available Elixir at least 6 on deployment | `flying_form_attack` | Inapplicable; deployment costs remain 3/6 |
| Hero Wizard | Flight, air damage, and pull | Paid ability | `enhanced_attack_area` | 1 |
| Hero Ice Golem | Air damage and slow | Paid ability | `moving_area` | 2 |
| Hero Giant | Selected air troop damage, throw, and stun | Paid ability | `selected_troop_ground_only_splash` | 2 |
| Mighty Miner | Air damage and knockback | Paid ability | `departure_bomb_area` | 1 |
| Monk | Eligible incoming projectile reflection | Paid ability | `eligible_incoming_projectiles` | 1 |

The optional flags `conditional_airborne`, `conditional_air_damage`,
`conditional_air_control`, and `conditional_air_reflection` describe potential
channels. `conditional_air_trigger` accepts `available_elixir_at_least_6` or
`paid_ability`. `conditional_air_response_scope` accepts the six values above;
`conditional_air_control_kind` accepts `pull`, `slow`, `throw_and_stun`, or
`knockback`. Omitted capabilities remain unknown, including Monk's direct air
damage. Reflection is its own restricted channel, not a universal air answer.
Hero Wizard's conditional attack augments his existing ordinary air targeting;
it must not add another card-level air-answer count. Deck identities cannot
establish deployment form, ability use, valid targets, or remaining ability uses.

Structural effects and Hero costs use the original developer-build observations:
[Spirit Empress](https://royaleapi.com/blog/spirit-empress-new-card-2025-july?lang=en),
[Wizard](https://royaleapi.com/blog/hero-wizard-january-2026),
[Ice Golem](https://royaleapi.com/blog/hero-ice-golem-january-2026), and
[Giant](https://royaleapi.com/blog/hero-giant-december-2025?lang=en).
The Giant's stun is corroborated by the
[official update](https://supercell.com/en/games/clashroyale/blog/release-notes/december-update-2025/).
Giant's landing splash affects ground troops only, even when the selected troop
was flying. Ice Golem's current `slow` uses the
[August 4 rework](https://supercell.com/en/games/clashroyale/blog/news/final-august-balance-changes-826/);
its earlier [knockback removal](https://supercell.com/en/games/clashroyale/blog/release-notes/april26-balance-changes/)
also applies. Original freeze/knockback descriptions are not current effects.

Mighty Miner's bomb air eligibility and knockback use explicit community
[Wiki observations](https://clashroyale.fandom.com/wiki/Mighty_Miner), reviewed
October 2; their field date is the review date. Its 1-Elixir ability cost uses the
[official April 8, 2022 change](https://supercell.com/en/games/clashroyale/blog/release-notes/balance-changes-april-2022/).
Monk's reflection toward the projectile's source uses the
[original gameplay preview](https://royaleapi.com/blog/monk-new-card?lang=en)
and [official description](https://supercell.com/en/games/clashroyale/blog/release-notes/new-update-october-2022/).
The [December 2025 Firecracker exception](https://supercell.com/en/games/clashroyale/blog/release-notes/december-update-2025/)
shows why eligible projectile reflection is restricted. Monk's ability cost uses
the [official December 2022 update](https://supercell.com/en/games/clashroyale/blog/release-notes/balance-changes-2).

For all five paid-ability candidates, `ability_usage: single_use_per_deployment`
uses the [August 4, 2026 rule](https://supercell.com/en/games/clashroyale/blog/news/final-august-balance-changes-826/),
which applies to Heroes and Champions except Boss Bandit. Older cooldown text is
superseded. Spirit Empress has no paid ability: `ability_usage` and `ability_cost`
are inapplicable, separate from her conditional deployment cost. No damage,
duration, range, or activation-frequency measurements are added. These catalog
fields do not alter the ordinary response predicates or strict static air audit.

The seventh revision records Spirit Empress's deployed actor as a troop
(`spell: false`), using the original preview's troop classification. Its internal
spell entry selects which troop form to deploy; this does not make the deployed
actor a direct spell response. The sixth revision remains frozen. This closes
the classification evidence needed by the conditional-aware hybrid audit.

The partial catalog is a preparation input. Freeze reviewed inputs and regenerate
schemas/caches before model comparison; retain historical catalog snapshots and
experiment artifacts. Spawned channels, ability activation, costs, and numerical
combat measurements still require their own evidence.
