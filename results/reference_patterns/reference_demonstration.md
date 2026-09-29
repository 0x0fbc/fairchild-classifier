# Reference-pattern demonstration

This demonstration applies Appendix C's provisional templates (Table C2) and registry (Table C3), unchanged, not piloted and not frozen, to characteristic labels, with the codebook operationalization departures listed below. The reference is `claude-opus-5-5`'s labels, with `gemini-3-1-pro-preview` as comparison; the classifier is the dev model at its calibration thresholds. No clarification probes, relevance or sufficiency component exist, so every answer is treated as relevant and sufficient, every record has prompt count 0, and no record is Invalid or insufficient. Allocation percentages summarise template matches in synthetic answers and do not estimate diagnostic probability.

## Templates

| Template | Name | Codebook conjunction | Registered domains | Note |
| --- | --- | --- | --- | --- |
| A1 | Interpretive difficulty with concern | Concern for welfare + Explicit interpretive difficulty + Acknowledged harm | Affective domain: concern-related reasoning, Cognitive domain: use of mental-state information | – |
| A2 | Systematic principle with acknowledged concern | Concern for welfare + Acknowledged harm + Consistent decision procedure | Affective domain: concern-related reasoning, Systemizing-related domain: rule justification | RULE is implied: consistent decision procedure is one of RULE's codebook labels. |
| A3 | Reflective harm appraisal | Concern for welfare + Acknowledged harm + Regret about harm + Consistent decision procedure + (Intention to repair or Acknowledgment of responsibility) | Post-decisional appraisal | – |
| B1 | Understood harm with explicit disregard | Explicit welfare disregard + Acknowledged harm | Affective domain: concern-related reasoning | – |
| B2 | Exploitative use of mental-state knowledge | Explicit welfare disregard + Exploitative use of understanding + Acknowledged harm + (Contextual mental-state explanation or Mental-state attribution) | Cognitive domain: use of mental-state information | AGENT_PERSPECTIVE is not coded, so only AFFECTED_PERSPECTIVE (CE1 or CE2) meets the perspective requirement. |
| B3 | Instrumental norm compliance | Explicit welfare disregard + Acknowledged harm + Instrumental rule compliance | Systemizing-related domain: rule justification | SELF_GAIN_PRIORITY and SELF_PROTECTION are not coded; instrumental rule compliance (followed only while it secures an advantage or prevents a cost) is taken to state one of them. |
| B4 | Self-consequence appraisal without harm regret | Explicit welfare disregard + Acknowledged harm + Explicit absence of harm regret + Regret about personal consequences | Post-decisional appraisal | – |

All conjunctions are checked at the answer level: raters and classifier mark each characteristic present or absent for the whole answer, so the same-person and same-time links of Table C2 are not verified.

## Registry

| Domain | Eligible questions | A templates | B templates | Test questions | Calibration questions | Contrast-set questions |
| --- | --- | --- | --- | --- | --- | --- |
| Affective domain: concern-related reasoning | Q1, Q3, Q4, Q5, Q6, Q8, Q10, Q12, Q14, Q15 | A1, A2 | B1 | Q3, Q6, Q14 | Q5 | Q1, Q3, Q4, Q5, Q6, Q8, Q10, Q12, Q14, Q15 |
| Cognitive domain: use of mental-state information | Q9, Q10, Q11, Q12, Q13 | A1 | B2 | Q11 | Q13 | Q9, Q10, Q11, Q12, Q13 |
| Personal distress (descriptive only) | Q1, Q4, Q5, Q6, Q7, Q8, Q11, Q12, Q15 | – | – | Q6, Q11 | Q5 | Q1, Q4, Q5, Q6, Q8, Q11, Q12, Q15 |
| Systemizing-related domain: rule justification | Q1, Q2, Q3, Q4, Q5, Q6, Q9, Q10, Q11, Q12, Q13 | A2 | B3 | Q3, Q6, Q11 | Q5, Q13 | Q1, Q2, Q3, Q4, Q5, Q6, Q9, Q10, Q11, Q12, Q13 |
| Post-decisional appraisal | Q8, Q12, Q15 | A3 | B4 | – | – | Q8, Q12, Q15 |

## Engine checks

| Check | Records | Expected | Computed |
| --- | --- | --- | --- |
| C6 example | A-only: 3, B-only: 1, Mixed: 2, Indeterminate: neither pattern: 1, Indeterminate: insufficient reasoning: 0, Invalid: 1, Processing review: 0, Missing or declined: 0 | N 8, D 6, A 66.7, B 33.3, Indeterminate 12.5, Invalid 12.5, coverage 75.0, review 0.0, missing 0.0 | N 8, D 6, A 66.7, B 33.3, Indeterminate 12.5, Invalid 12.5, coverage 75.0, review 0.0, missing 0.0 |
| All Indeterminate | A-only: 0, B-only: 0, Mixed: 0, Indeterminate: neither pattern: 8, Indeterminate: insufficient reasoning: 0, Invalid: 0, Processing review: 0, Missing or declined: 0 | N 8, D 0, A –, B –, Indeterminate 100.0, Invalid 0.0, coverage 0.0, review 0.0, missing 0.0 | N 8, D 0, A –, B –, Indeterminate 100.0, Invalid 0.0, coverage 0.0, review 0.0, missing 0.0 |
| All Mixed | A-only: 0, B-only: 0, Mixed: 8, Indeterminate: neither pattern: 0, Indeterminate: insufficient reasoning: 0, Invalid: 0, Processing review: 0, Missing or declined: 0 | N 8, D 8, A 50.0, B 50.0, Indeterminate 0.0, Invalid 0.0, coverage 100.0, review 0.0, missing 0.0 | N 8, D 8, A 50.0, B 50.0, Indeterminate 0.0, Invalid 0.0, coverage 100.0, review 0.0, missing 0.0 |
| All Invalid | A-only: 0, B-only: 0, Mixed: 0, Indeterminate: neither pattern: 0, Indeterminate: insufficient reasoning: 0, Invalid: 8, Processing review: 0, Missing or declined: 0 | N 8, D 0, A –, B –, Indeterminate 0.0, Invalid 100.0, coverage 0.0, review 0.0, missing 0.0 | N 8, D 0, A –, B –, Indeterminate 0.0, Invalid 100.0, coverage 0.0, review 0.0, missing 0.0 |
| Empty | A-only: 0, B-only: 0, Mixed: 0, Indeterminate: neither pattern: 0, Indeterminate: insufficient reasoning: 0, Invalid: 0, Processing review: 0, Missing or declined: 0 | N 0, D 0, A –, B –, Indeterminate –, Invalid –, coverage –, review –, missing – | N 0, D 0, A –, B –, Indeterminate –, Invalid –, coverage –, review –, missing – |

In every check the displayed counts sum to N.

## Contrast set

Briefs target full conjunctions and near misses on each template's registered questions. A1 substitutes contextual mental-state explanation for mentalizing difficulty; A2 substitutes ordinary rule justification for a consistent procedure; A3 includes responsibility and repair variants, with procedure omitted in their near misses. B1's near miss uses consequence reasoning without concern or disregard; B2 contrasts exploitative use with contextual consequence reasoning; B3 contrasts disregard with concern while retaining instrumental norm use; B4 contrasts explicit absence of harm regret with harm regret while retaining self-consequence regret. B2 and B4 each have two positive and two near-miss briefs per eligible question. Mixed briefs require a consistent procedure and concern for one identified person alongside disregard for another, following Table C5's Q05 example. These are separately generated answers, not paired edits; some near misses remove more than one element.

The generators were `claude-opus-5-5` at medium effort and `claude-sonnet-5` at default effort, one batch of 10 answers per brief per generator. Both raters used the standard characteristic-annotation prompt, without design labels or direct template-level annotation. Declined batches were excluded and not re-prompted.

Total: 150 briefs, 300 batches, 0 declined batches and 3000 answers.

| Template | Case | Briefs | Batches | Declined | Answers | Rater-1 agreement with design |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | positive | 13 | 26 | 0 | 260 | 0.615 |
| A1 | near_miss | 13 | 26 | 0 | 260 | 0.996 |
| A2 | positive | 14 | 28 | 0 | 280 | 0.500 |
| A2 | near_miss | 14 | 28 | 0 | 280 | 1.000 |
| A3 | positive | 6 | 12 | 0 | 120 | 0.275 |
| A3 | near_miss | 6 | 12 | 0 | 120 | 1.000 |
| B1 | positive | 10 | 20 | 0 | 200 | 0.720 |
| B1 | near_miss | 10 | 20 | 0 | 200 | 1.000 |
| B2 | positive | 10 | 20 | 0 | 200 | 0.945 |
| B2 | near_miss | 10 | 20 | 0 | 200 | 1.000 |
| B3 | positive | 11 | 22 | 0 | 220 | 0.841 |
| B3 | near_miss | 11 | 22 | 0 | 220 | 1.000 |
| B4 | positive | 6 | 12 | 0 | 120 | 0.458 |
| B4 | near_miss | 6 | 12 | 0 | 120 | 1.000 |
| A2+B1 | mixed | 10 | 20 | 0 | 200 | 0.230 |

Generation cost: claude-opus-5-5 $10.95; claude-sonnet-5 $3.97. Annotation cost: claude-opus-5-5 $14.80; gemini-3-1-pro-preview $8.98. Total saved-completion cost: $38.70.

## Outcome agreement

Agreement and κ use ungated outcomes against rater 1. Support F1 intervals resample whole briefs (2,000 draws; 95% percentile intervals). F1 is undefined when the reference has no positive or no negative. Review uses a ±0.10 shift of every threshold, clipped to 0.01–0.99; resolved agreement excludes review. All-domains rows pool eligible answer–domain records, not independent answers. All shares are fractions.

### Test questions

| Domain | Records | Rater-1 A-only / B-only / Mixed / Indeterminate | Classifier agreement | Classifier κ | Support A F1 (95% CI) | Support B F1 (95% CI) | Review share | Agreement when resolved | Rater-2 agreement | Rater-2 κ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Affective domain: concern-related reasoning | 7120 | 38 / 51 / 0 / 7031 | 0.994 | 0.755 | 0.594 (0.000–0.767) | 0.850 (0.784–0.905) | 0.003 | 0.995 | 0.992 | 0.683 |
| Cognitive domain: use of mental-state information | 2360 | 5 / 0 / 0 / 2355 | 0.997 | 0.249 | 0.250 (0.000–0.667) | – | 0.000 | 0.998 | 0.997 | -0.001 |
| Systemizing-related domain: rule justification | 7080 | 21 / 0 / 0 / 7059 | 0.996 | 0.314 | 0.316 (0.000–0.500) | – | 0.001 | 0.997 | 0.997 | 0.443 |
| Post-decisional appraisal | 0 | 0 / 0 / 0 / 0 | – | – | – | – | – | – | – | – |
| **All domains** | 16560 | 64 / 51 / 0 / 16445 | 0.995 | – | – | – | 0.002 | 0.996 | 0.995 | – |

| Domain | Rater-2 support A F1 (95% CI) | Rater-2 support B F1 (95% CI) |
| --- | --- | --- |
| Affective domain: concern-related reasoning | 0.320 (0.000–0.486) | 0.836 (0.808–0.857) |
| Cognitive domain: use of mental-state information | 0.000 (0.000–0.000) | – |
| Systemizing-related domain: rule justification | 0.444 (0.000–0.727) | – |
| Post-decisional appraisal | – | – |

### Calibration questions

Thresholds were tuned on these questions, so these scores are optimistic.

| Domain | Records | Rater-1 A-only / B-only / Mixed / Indeterminate | Classifier agreement | Classifier κ | Support A F1 (95% CI) | Support B F1 (95% CI) | Review share | Agreement when resolved | Rater-2 agreement | Rater-2 κ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Affective domain: concern-related reasoning | 2370 | 3 / 2 / 0 / 2365 | 0.995 | -0.002 | 0.000 (0.000–0.000) | 0.000 (0.000–0.000) | 0.002 | 0.996 | – | – |
| Cognitive domain: use of mental-state information | 2400 | 19 / 0 / 0 / 2381 | 0.993 | 0.331 | 0.333 (0.000–0.667) | – | 0.001 | 0.994 | – | – |
| Systemizing-related domain: rule justification | 4770 | 8 / 0 / 0 / 4762 | 0.999 | 0.363 | 0.364 (0.000–0.444) | – | 0.000 | 0.999 | – | – |
| Post-decisional appraisal | 0 | 0 / 0 / 0 / 0 | – | – | – | – | – | – | – | – |
| **All domains** | 9540 | 30 / 2 / 0 / 9508 | 0.996 | – | – | – | 0.001 | 0.997 | – | – |

| Domain | Rater-2 support A F1 (95% CI) | Rater-2 support B F1 (95% CI) |
| --- | --- | --- |
| Affective domain: concern-related reasoning | – | – |
| Cognitive domain: use of mental-state information | – | – |
| Systemizing-related domain: rule justification | – | – |
| Post-decisional appraisal | – | – |

### Contrast set

| Domain | Records | Rater-1 A-only / B-only / Mixed / Indeterminate | Classifier agreement | Classifier κ | Support A F1 (95% CI) | Support B F1 (95% CI) | Review share | Agreement when resolved | Rater-2 agreement | Rater-2 κ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Affective domain: concern-related reasoning | 2320 | 351 / 461 / 46 / 1462 | 0.881 | 0.774 | 0.785 (0.734–0.826) | 0.866 (0.818–0.904) | 0.090 | 0.905 | 0.890 | 0.794 |
| Cognitive domain: use of mental-state information | 1280 | 77 / 190 / 0 / 1013 | 0.959 | 0.878 | 0.880 (0.741–0.936) | 0.903 (0.845–0.949) | 0.070 | 0.971 | 0.981 | 0.945 |
| Systemizing-related domain: rule justification | 2260 | 216 / 199 / 0 / 1845 | 0.932 | 0.770 | 0.768 (0.713–0.819) | 0.836 (0.739–0.907) | 0.051 | 0.943 | 0.944 | 0.821 |
| Post-decisional appraisal | 1020 | 34 / 55 / 0 / 931 | 0.953 | 0.666 | 0.594 (0.436–0.711) | 0.750 (0.583–0.900) | 0.030 | 0.964 | 0.961 | 0.748 |
| **All domains** | 6880 | 678 / 905 / 46 / 5251 | 0.923 | – | – | – | 0.065 | 0.939 | 0.935 | – |

| Domain | Rater-2 support A F1 (95% CI) | Rater-2 support B F1 (95% CI) |
| --- | --- | --- |
| Affective domain: concern-related reasoning | 0.784 (0.721–0.835) | 0.894 (0.841–0.936) |
| Cognitive domain: use of mental-state information | 0.867 (0.687–0.939) | 0.989 (0.978–0.998) |
| Systemizing-related domain: rule justification | 0.783 (0.708–0.850) | 0.907 (0.808–0.970) |
| Post-decisional appraisal | 0.556 (0.389–0.769) | 0.862 (0.739–0.983) |

## Templates against rater 1

### Test questions

| Template | Answers assessed | Rater-1 matches | Classifier matches | Classifier F1 (95% CI) | Rater-2 F1 (95% CI) |
| --- | --- | --- | --- | --- | --- |
| A1 | 9480 | 31 | 25 | 0.643 (0.000–0.774) | 0.372 (0.000–0.478) |
| A2 | 9480 | 33 | 19 | 0.308 (0.000–0.462) | 0.333 (0.000–0.667) |
| A3 | 0 | 0 | 0 | – | – |
| B1 | 7120 | 51 | 62 | 0.850 (0.784–0.905) | 0.836 (0.808–0.857) |
| B2 | 2360 | 0 | 0 | – | – |
| B3 | 7080 | 0 | 0 | – | – |
| B4 | 0 | 0 | 0 | – | – |

### Contrast set

| Template | Answers assessed | Rater-1 matches | Classifier matches | Classifier F1 (95% CI) | Rater-2 F1 (95% CI) |
| --- | --- | --- | --- | --- | --- |
| A1 | 2920 | 172 | 174 | 0.855 (0.792–0.903) | 0.791 (0.687–0.861) |
| A2 | 3000 | 321 | 276 | 0.754 (0.702–0.801) | 0.778 (0.707–0.837) |
| A3 | 1020 | 34 | 30 | 0.594 (0.436–0.711) | 0.556 (0.389–0.769) |
| B1 | 2320 | 507 | 445 | 0.866 (0.818–0.904) | 0.894 (0.841–0.936) |
| B2 | 1280 | 190 | 162 | 0.903 (0.845–0.949) | 0.989 (0.978–0.998) |
| B3 | 2260 | 199 | 160 | 0.836 (0.739–0.907) | 0.907 (0.808–0.970) |
| B4 | 1020 | 55 | 33 | 0.750 (0.583–0.900) | 0.862 (0.739–0.983) |

## Contrast-set confusion

### Affective domain: concern-related reasoning

| Rater 1 \ Classifier | A-only | B-only | Mixed | Indeterminate | Processing review |
| --- | --- | --- | --- | --- | --- |
| A-only | 254 | 0 | 0 | 53 | 44 |
| B-only | 0 | 313 | 0 | 46 | 102 |
| Mixed | 7 | 14 | 3 | 9 | 13 |
| Indeterminate | 50 | 21 | 1 | 1340 | 50 |

### Cognitive domain: use of mental-state information

| Rater 1 \ Classifier | A-only | B-only | Mixed | Indeterminate | Processing review |
| --- | --- | --- | --- | --- | --- |
| A-only | 63 | 0 | 0 | 11 | 3 |
| B-only | 0 | 104 | 0 | 16 | 70 |
| Mixed | 0 | 0 | 0 | 0 | 0 |
| Indeterminate | 7 | 0 | 0 | 990 | 16 |

### Systemizing-related domain: rule justification

| Rater 1 \ Classifier | A-only | B-only | Mixed | Indeterminate | Processing review |
| --- | --- | --- | --- | --- | --- |
| A-only | 140 | 0 | 0 | 47 | 29 |
| B-only | 0 | 104 | 0 | 37 | 58 |
| Mixed | 0 | 0 | 0 | 0 | 0 |
| Indeterminate | 31 | 8 | 0 | 1777 | 29 |

### Post-decisional appraisal

| Rater 1 \ Classifier | A-only | B-only | Mixed | Indeterminate | Processing review |
| --- | --- | --- | --- | --- | --- |
| A-only | 16 | 0 | 0 | 13 | 5 |
| B-only | 0 | 21 | 0 | 15 | 19 |
| Mixed | 0 | 0 | 0 | 0 | 0 |
| Indeterminate | 8 | 0 | 0 | 916 | 7 |

## Contrast set by question familiarity

| Questions | Records | Classifier agreement | Review share | Agreement when resolved | Rater-2 agreement |
| --- | --- | --- | --- | --- | --- |
| Training questions (new answers) | 4860 | 0.930 | 0.057 | 0.945 | 0.937 |
| Calibration questions | 760 | 0.925 | 0.084 | 0.934 | 0.937 |
| Test questions | 1260 | 0.897 | 0.082 | 0.919 | 0.927 |

## Indeterminate rates

### Held-out corpus

| Domain | Question | Records | Rater-1 Indeterminate | Classifier Indeterminate | Review share |
| --- | --- | --- | --- | --- | --- |
| Affective domain: concern-related reasoning | Q3 | 2340 | 1.000 | 0.999 | 0.001 |
| Affective domain: concern-related reasoning | Q5 | 2370 | 0.998 | 0.997 | 0.002 |
| Affective domain: concern-related reasoning | Q6 | 2380 | 0.991 | 0.990 | 0.003 |
| Affective domain: concern-related reasoning | Q14 | 2400 | 0.972 | 0.974 | 0.004 |
| Cognitive domain: use of mental-state information | Q11 | 2360 | 0.998 | 0.999 | 0.000 |
| Cognitive domain: use of mental-state information | Q13 | 2400 | 0.992 | 0.998 | 0.001 |
| Systemizing-related domain: rule justification | Q3 | 2340 | 1.000 | 1.000 | 0.000 |
| Systemizing-related domain: rule justification | Q5 | 2370 | 1.000 | 1.000 | 0.000 |
| Systemizing-related domain: rule justification | Q6 | 2380 | 1.000 | 0.999 | 0.000 |
| Systemizing-related domain: rule justification | Q11 | 2360 | 0.991 | 0.994 | 0.003 |
| Systemizing-related domain: rule justification | Q13 | 2400 | 0.997 | 0.999 | 0.000 |

### Contrast set

| Domain | Question | Records | Rater-1 Indeterminate | Classifier Indeterminate | Review share |
| --- | --- | --- | --- | --- | --- |
| Affective domain: concern-related reasoning | Q1 | 180 | 0.594 | 0.583 | 0.039 |
| Affective domain: concern-related reasoning | Q3 | 180 | 0.600 | 0.606 | 0.078 |
| Affective domain: concern-related reasoning | Q4 | 180 | 0.828 | 0.861 | 0.072 |
| Affective domain: concern-related reasoning | Q5 | 180 | 0.589 | 0.600 | 0.122 |
| Affective domain: concern-related reasoning | Q6 | 180 | 0.661 | 0.728 | 0.133 |
| Affective domain: concern-related reasoning | Q8 | 300 | 0.627 | 0.633 | 0.090 |
| Affective domain: concern-related reasoning | Q10 | 260 | 0.535 | 0.546 | 0.069 |
| Affective domain: concern-related reasoning | Q12 | 420 | 0.636 | 0.726 | 0.105 |
| Affective domain: concern-related reasoning | Q14 | 140 | 0.650 | 0.629 | 0.071 |
| Affective domain: concern-related reasoning | Q15 | 300 | 0.627 | 0.667 | 0.100 |
| Cognitive domain: use of mental-state information | Q9 | 200 | 0.705 | 0.730 | 0.105 |
| Cognitive domain: use of mental-state information | Q10 | 260 | 0.788 | 0.796 | 0.038 |
| Cognitive domain: use of mental-state information | Q11 | 200 | 0.740 | 0.790 | 0.120 |
| Cognitive domain: use of mental-state information | Q12 | 420 | 0.886 | 0.910 | 0.045 |
| Cognitive domain: use of mental-state information | Q13 | 200 | 0.735 | 0.760 | 0.075 |
| Systemizing-related domain: rule justification | Q1 | 180 | 0.772 | 0.789 | 0.028 |
| Systemizing-related domain: rule justification | Q2 | 80 | 0.587 | 0.587 | 0.025 |
| Systemizing-related domain: rule justification | Q3 | 180 | 0.778 | 0.817 | 0.056 |
| Systemizing-related domain: rule justification | Q4 | 180 | 0.867 | 0.894 | 0.061 |
| Systemizing-related domain: rule justification | Q5 | 180 | 0.728 | 0.772 | 0.078 |
| Systemizing-related domain: rule justification | Q6 | 180 | 0.794 | 0.878 | 0.078 |
| Systemizing-related domain: rule justification | Q9 | 200 | 0.855 | 0.855 | 0.050 |
| Systemizing-related domain: rule justification | Q10 | 260 | 0.838 | 0.838 | 0.035 |
| Systemizing-related domain: rule justification | Q11 | 200 | 0.895 | 0.870 | 0.035 |
| Systemizing-related domain: rule justification | Q12 | 420 | 0.824 | 0.890 | 0.050 |
| Systemizing-related domain: rule justification | Q13 | 200 | 0.875 | 0.880 | 0.065 |
| Post-decisional appraisal | Q8 | 300 | 0.923 | 0.920 | 0.040 |
| Post-decisional appraisal | Q12 | 420 | 0.917 | 0.957 | 0.021 |
| Post-decisional appraisal | Q15 | 300 | 0.897 | 0.930 | 0.033 |

On test questions, rater-1 Indeterminate records were 16445 / 16560 = 0.993 (99.3%).

There were no probes (prompt count 0 throughout) and no Invalid records. Insufficient reasoning was not assessed, so there were no insufficient-reasoning records.

## Personal distress

Descriptive only; no A/B allocation or directional Indeterminate outcome is defined. Rater 2 in the held-out corpus covers test answers only, not calibration answers. Cells are prevalence shares.

| Source | Records | DISTRESS expressed: rater 1 / classifier / rater 2 | Explicitly low distress: rater 1 / classifier / rater 2 |
| --- | --- | --- | --- |
| Held-out corpus | 7110 | 0.490 / 0.485 / 0.420 | 0.092 / 0.095 / 0.088 |
| Contrast set | 1940 | 0.405 / 0.410 / 0.385 | 0.008 / 0.009 / 0.004 |

## Aggregated administrations

Each of 1,000 synthetic administrations draws one answer per question and does not represent a respondent. The held-out corpus draws from Q3, Q5, Q6, Q11, Q13 and Q14; the contrast set draws from every question it contains, all but Q7. Draws use seed 1, one random stream per source, questions in numeric order and answer candidates sorted by id. Each applicable domain uses unit weights. No minimum reporting coverage was set. D = 0 cells are shares of administrations; allocation differences are percentage points, restricted to administrations where both reports have D > 0.

| Source | Domain | Items (N) | D = 0: rater 1 / classifier | Mean coverage %: rater 1 / classifier | Mean A allocation % (D > 0): rater 1 / classifier | Mean absolute A-allocation difference | Administrations with both D > 0 | Mean review % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Held-out corpus | Affective domain: concern-related reasoning | 4 | 0.962 / 0.968 | 0.9 / 0.8 | 50.0 / 31.2 | 0.0 | 22 | 0.2 |
| Held-out corpus | Cognitive domain: use of mental-state information | 2 | 0.989 / 0.995 | 0.6 / 0.2 | 100.0 / 100.0 | 0.0 | 5 | 0.1 |
| Held-out corpus | Systemizing-related domain: rule justification | 5 | 0.985 / 0.993 | 0.3 / 0.1 | 100.0 / 100.0 | 0.0 | 2 | 0.1 |
| Held-out corpus | Post-decisional appraisal | not assessed | – | – | – | – | – | – |
| Contrast set | Affective domain: concern-related reasoning | 10 | 0.007 / 0.033 | 37.0 / 29.6 | 47.7 / 49.2 | 13.6 | 966 | 8.7 |
| Contrast set | Cognitive domain: use of mental-state information | 5 | 0.269 / 0.436 | 22.7 / 15.0 | 28.8 / 37.8 | 6.8 | 555 | 7.1 |
| Contrast set | Systemizing-related domain: rule justification | 11 | 0.083 / 0.163 | 20.2 / 14.6 | 50.1 / 56.6 | 14.1 | 819 | 4.9 |
| Contrast set | Post-decisional appraisal | 3 | 0.754 / 0.864 | 9.0 / 4.8 | 42.5 / 55.9 | 3.8 | 120 | 3.3 |

### Example administration

The first contrast-set administration:

| Domain | Report | N | A-only | B-only | Mixed | Indeterminate (neither / insufficient) | Invalid | Review | Missing | D | A allocation | B allocation | Indeterminate % | Invalid % | Coverage % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Affective domain: concern-related reasoning | Rater 1 | 10 | 4 | 1 | 0 | 5 / 0 | 0 | 0 | 0 | 5 | 80.0 | 20.0 | 50.0 | 0.0 | 50.0 |
| Affective domain: concern-related reasoning | Classifier | 10 | 4 | 0 | 0 | 6 / 0 | 0 | 0 | 0 | 4 | 100.0 | 0.0 | 60.0 | 0.0 | 40.0 |
| Cognitive domain: use of mental-state information | Rater 1 | 5 | 1 | 2 | 0 | 2 / 0 | 0 | 0 | 0 | 3 | 33.3 | 66.7 | 40.0 | 0.0 | 60.0 |
| Cognitive domain: use of mental-state information | Classifier | 5 | 1 | 1 | 0 | 2 / 0 | 0 | 1 | 0 | 2 | 50.0 | 50.0 | 40.0 | 0.0 | 40.0 |
| Systemizing-related domain: rule justification | Rater 1 | 11 | 1 | 1 | 0 | 9 / 0 | 0 | 0 | 0 | 2 | 50.0 | 50.0 | 81.8 | 0.0 | 18.2 |
| Systemizing-related domain: rule justification | Classifier | 11 | 1 | 0 | 0 | 9 / 0 | 0 | 1 | 0 | 1 | 100.0 | 0.0 | 81.8 | 0.0 | 9.1 |
| Post-decisional appraisal | Rater 1 | 3 | 0 | 0 | 0 | 3 / 0 | 0 | 0 | 0 | 0 | – | – | 100.0 | 0.0 | 0.0 |
| Post-decisional appraisal | Classifier | 3 | 0 | 0 | 0 | 3 / 0 | 0 | 0 | 0 | 0 | – | – | 100.0 | 0.0 | 0.0 |

## Limits

- Answers and both raters are language models, not participants or independently adjudicated human annotations. Agreement is with rater 1, not clinical ground truth.
- The registry remains provisional: it was applied without piloting, review or freezing. Calibration scores are optimistic; contrast answers to training questions are new answers, not new dilemmas.
- Conjunctions are answer-level: same-person and same-time links, coherent scope and own-action scope are not verified by binary characteristic labels. The A2, B2 and B3 operationalization departures are stated in the Templates table.
- There are no probes, relevance or sufficiency components. Invalid and insufficient outcomes are supported by the aggregation engine, not detected in these answers. Review reflects threshold sensitivity, not a validated clinical uncertainty rule.
- Contrast-set rates reflect the targeted design and excluded declines, not prevalence in any population. Synthetic administrations mix independent answers and are not respondents.
- Allocation percentages are not probabilities of ASD or ASPD, and this demonstration cannot establish clinical differentiation.
