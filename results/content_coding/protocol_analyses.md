# Protocol analyses

These are the analyses the concept paper's synthetic-evaluation protocol (Sections 4.10–4.12) asks for beyond `results/content_coding/classifier_evaluation.md`. They use the dev model's test predictions at its calibration-tuned thresholds (`results/content_coding/predictions.jsonl`), with `claude-opus-5-5`'s labels as the reference and `gemini-3-1-pro-preview` as the second rater. Intervals are 95% percentile intervals from 2,000 resamples of the 240 test briefs, the paper's parent groups, with all answers to a brief resampled together.

## Core codes

The paper's core codes (Section 4.9) as unions of codebook labels: a code is present in an answer when any of its labels is. AGENT_PERSPECTIVE has no codebook label and is not scored. Precision, recall, F1 and κ (Cohen's kappa) compare the classifier's predictions and the second rater's labels with `claude-opus-5-5`'s.

| Code | Codebook labels | Answers | `claude-opus-5-5` positives | Precision | Recall | Classifier F1 (95% CI) | Classifier κ | Rater F1 (95% CI) | Rater κ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RULE | Rule-based justification, Consistent decision procedure, Fairness and reciprocity | 9,480 | 3,709 | 0.895 | 0.892 | 0.893 (0.873–0.913) | 0.825 | 0.899 (0.877–0.919) | 0.833 |
| CONSEQUENCE | Consequence-based justification | 9,480 | 2,465 | 0.864 | 0.802 | 0.832 (0.807–0.853) | 0.776 | 0.796 (0.758–0.826) | 0.713 |
| CONCERN | Concern for welfare | 9,480 | 1,086 | 0.832 | 0.675 | 0.745 (0.666–0.798) | 0.716 | 0.697 (0.594–0.765) | 0.663 |
| DISTRESS | Distress at another’s harm, Distress from decision conflict, Anxiety about personal consequences | 9,480 | 4,311 | 0.875 | 0.891 | 0.883 (0.862–0.901) | 0.784 | 0.844 (0.813–0.871) | 0.721 |
| AFFECTED_PERSPECTIVE | Mental-state attribution, Contextual mental-state explanation | 9,480 | 3,203 | 0.911 | 0.899 | 0.905 (0.885–0.921) | 0.857 | 0.805 (0.756–0.844) | 0.724 |
| **Macro average** | – | – | – | 0.876 | 0.832 | 0.852 (0.835–0.864) | 0.792 | 0.808 (0.786–0.825) | 0.731 |

## Test questions by story setting

Story settings group the questions set in the same story. Every test question is held out whole. Under the paper's Table 5 content families, all four test questions share a family with a training question, so this is not the family holdout the protocol specifies. It separates the test questions whose story setting appears in training from those whose setting does not. Macro AP uses the saved probabilities, which are rounded to four decimals; the ties this creates lower it slightly against the main report.

| Story setting | Train | Calibration | Test |
|---|---|---|---|
| Railway | Q1, Q2, Q4, Q7, Q8 | Q5 | Q3 |
| Island rescue | – | – | Q6 |
| Warehouse | Q9, Q10 | – | Q11 |
| Shared paid job | Q12, Q15 | – | – |
| Game exchange | – | Q13 | – |
| Coworker reward | – | – | Q14 |

| Test questions | Questions | Answers | Classifier macro F1 (95% CI) | Macro AP | All correct | Second-rater macro F1 (95% CI) |
|---|---|---:|---:|---:|---:|---:|
| Setting seen in training | Q3, Q11 | 4,700 | 0.876 (0.858–0.883) | 0.934 | 0.451 | 0.869 (0.847–0.877) |
| New setting | Q6, Q14 | 4,780 | 0.826 (0.809–0.843) | 0.884 | 0.376 | 0.812 (0.787–0.830) |
| All test questions | Q3, Q6, Q11, Q14 | 9,480 | 0.852 (0.837–0.862) | 0.903 | 0.413 | 0.840 (0.822–0.852) |

## Test by answer style

Each brief set one approach and one tone for its answers; the test answers are grouped by their brief's.

| Answer style | Answers | Classifier macro F1 | All correct | Second-rater macro F1 |
|---|---:|---:|---:|---:|
| Approach: Deliberative | 3,220 | 0.836 | 0.365 | 0.820 |
| Approach: Intuitive | 3,100 | 0.818 | 0.397 | 0.802 |
| Approach: Avoidant | 3,160 | 0.820 | 0.478 | 0.808 |
| Tone: Calm | 2,370 | 0.866 | 0.512 | 0.859 |
| Tone: Energized | 2,370 | 0.858 | 0.464 | 0.850 |
| Tone: Distressed | 2,360 | 0.801 | 0.364 | 0.780 |
| Tone: Dejected | 2,380 | 0.794 | 0.314 | 0.799 |

## Confidence intervals

Intervals resample the test briefs as described above. The classifier and the second rater are compared with `claude-opus-5-5` on the same resamples, so their difference is paired.

| Measure | Value (95% CI) |
|---|---:|
| Classifier macro F1 | 0.852 (0.837–0.862) |
| Second-rater macro F1 | 0.840 (0.822–0.852) |
| Classifier minus second-rater macro F1 (paired) | +0.012 (+0.004–+0.021) |
| Classifier all correct | 0.413 (0.389–0.439) |
| Second rater all agree | 0.391 (0.362–0.421) |
| Classifier pair accuracy | 0.966 (0.964–0.968) |
| Core-code classifier macro F1 | 0.852 (0.835–0.864) |
| Core-code second-rater macro F1 | 0.808 (0.786–0.825) |

| Label | Positives | Classifier F1 (95% CI) | Rater F1 (95% CI) | Classifier − rater (95% CI) |
|---|---:|---:|---:|---:|
| Shared emotion | 708 | 0.961 (0.925–0.983) | 0.982 (0.953–0.997) | -0.021 (-0.041–-0.009) |
| Explicit emotional nonsharing | 788 | 0.989 (0.982–0.994) | 0.994 (0.989–0.998) | -0.005 (-0.010–-0.001) |
| Concern for welfare | 1,086 | 0.745 (0.666–0.798) | 0.697 (0.594–0.765) | +0.048 (+0.021–+0.087) |
| Explicit welfare disregard | 632 | 0.897 (0.843–0.938) | 0.908 (0.847–0.956) | -0.011 (-0.035–+0.014) |
| Mental-state attribution | 3,203 | 0.905 (0.885–0.921) | 0.805 (0.756–0.844) | +0.100 (+0.069–+0.137) |
| Contextual mental-state explanation | 757 | 0.662 (0.579–0.732) | 0.501 (0.370–0.611) | +0.161 (+0.094–+0.243) |
| Explicit interpretive difficulty | 930 | 0.913 (0.870–0.940) | 0.897 (0.837–0.931) | +0.016 (-0.002–+0.041) |
| Acknowledged harm | 1,673 | 0.773 (0.729–0.809) | 0.709 (0.638–0.766) | +0.065 (+0.026–+0.108) |
| Supportive use of understanding | 262 | 0.639 (0.422–0.745) | 0.640 (0.392–0.750) | -0.001 (-0.083–+0.106) |
| Exploitative use of understanding | 63 | 0.671 (0.440–0.892) | 0.678 (0.397–0.965) | -0.007 (-0.082–+0.040) |
| Distress at another’s harm | 1,272 | 0.812 (0.756–0.853) | 0.767 (0.687–0.822) | +0.044 (+0.009–+0.087) |
| Distress from decision conflict | 2,258 | 0.755 (0.718–0.790) | 0.704 (0.653–0.752) | +0.051 (+0.019–+0.084) |
| Anxiety about personal consequences | 1,288 | 0.843 (0.786–0.883) | 0.837 (0.778–0.880) | +0.007 (-0.023–+0.036) |
| Explicitly low distress | 896 | 0.948 (0.915–0.969) | 0.959 (0.923–0.979) | -0.011 (-0.025–+0.004) |
| Rule-based justification | 1,500 | 0.788 (0.736–0.829) | 0.756 (0.695–0.807) | +0.032 (-0.011–+0.077) |
| Consistent decision procedure | 773 | 0.926 (0.903–0.943) | 0.895 (0.842–0.929) | +0.032 (+0.004–+0.074) |
| Fairness and reciprocity | 2,165 | 0.827 (0.793–0.857) | 0.828 (0.792–0.860) | -0.001 (-0.022–+0.019) |
| Context-sensitive rule application | 825 | 0.909 (0.872–0.933) | 0.951 (0.925–0.966) | -0.042 (-0.071–-0.018) |
| Consequence-based justification | 2,465 | 0.832 (0.807–0.853) | 0.796 (0.758–0.826) | +0.036 (+0.013–+0.060) |
| Instrumental rule compliance | 728 | 0.958 (0.930–0.976) | 0.962 (0.926–0.987) | -0.004 (-0.023–+0.015) |
| Regret about harm | 663 | 0.853 (0.804–0.884) | 0.854 (0.801–0.890) | -0.002 (-0.039–+0.035) |
| Regret about personal consequences | 574 | 0.876 (0.818–0.915) | 0.871 (0.812–0.911) | +0.004 (-0.042–+0.055) |
| Explicit absence of harm regret | 713 | 0.901 (0.867–0.929) | 0.919 (0.875–0.956) | -0.019 (-0.055–+0.024) |
| Acknowledgment of responsibility | 1,421 | 0.841 (0.792–0.874) | 0.833 (0.774–0.873) | +0.008 (-0.016–+0.035) |
| Intention to repair | 539 | 0.707 (0.623–0.772) | 0.858 (0.785–0.912) | -0.151 (-0.206–-0.106) |
| Difficulty identifying own emotion | 814 | 0.988 (0.980–0.993) | 0.987 (0.975–0.994) | +0.001 (-0.005–+0.010) |
| Prediction of another’s regret | 744 | 0.967 (0.951–0.978) | 0.985 (0.971–0.993) | -0.017 (-0.031–-0.006) |
| Practical counterfactual | 823 | 0.957 (0.928–0.974) | 0.939 (0.892–0.966) | +0.018 (-0.005–+0.053) |

The interval of the difference lies wholly above zero for 8 labels, wholly below zero for 5 (Shared emotion, Explicit emotional nonsharing, Context-sensitive rule application, Intention to repair, Prediction of another’s regret), and includes zero for 15.

## Evidence quotes

Each rater quotes the shortest phrase, at most 15 words, that shows each characteristic it marks present. Verbatim means the quote occurs exactly in the answer. After normalisation means it occurs once both are normalised: NFKC, casefolding, collapsed whitespace, plain quote marks and dashes, and the quote's surrounding spaces, full stops, ellipses and quote marks stripped. An empty quote counts as not found. Mental-state attributions added only as a prerequisite of another label carry no quote and are not counted.

| Rater | Split | Answers | Quotes | Verbatim | After normalisation | Not found |
|---|---|---:|---:|---:|---:|---:|
| `claude-opus-5-5` | all | 35,930 | 121,350 | 0.991 | 0.991 | 1,069 |
| `claude-opus-5-5` | test | 9,480 | 30,449 | 0.990 | 0.990 | 306 |
| `gemini-3-1-pro-preview` | test | 9,480 | 29,270 | 0.975 | 0.977 | 683 |

Where both raters mark the same characteristic in a test answer, each quote is located at its first occurrence in the normalised answer; the two spans overlap when they share at least one character. Spans overlap is the share of the pairs with both quotes located.

| Domain | Pairs both raters mark | Both quotes located | Spans overlap |
|---|---:|---:|---:|
| Affective empathy | 2,777 | 2,702 | 0.950 |
| Cognitive empathy | 4,492 | 4,396 | 0.915 |
| Personal distress | 4,441 | 4,313 | 0.895 |
| Systemizing | 7,320 | 6,940 | 0.859 |
| Post-decision appraisal | 5,740 | 5,614 | 0.949 |
| **All** | 24,770 | 23,965 | 0.907 |

## Contrasting labels

The label pairs the paper asks to keep apart (Section 4.10). For the answers where `claude-opus-5-5` marks the first label but not the second: how often the classifier finds the first, how often it wrongly predicts the second, and how often `gemini-3-1-pro-preview` marks the second; then the same with the labels swapped. Unprovoked versus defensive noncooperation has no codebook labels.

| Contrast | Only first: answers | First found | Second predicted | Rater marks second | Only second: answers | Second found | First predicted | Rater marks first |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *Concern for welfare* vs *Shared emotion* | 995 | 0.659 | 0.010 | 0.004 | 617 | 0.997 | 0.005 | 0.021 |
| *Regret about harm* vs *Regret about personal consequences* | 621 | 0.899 | 0.026 | 0.074 | 532 | 0.846 | 0.039 | 0.023 |
| *Acknowledged harm* vs *Explicit welfare disregard* | 1,611 | 0.796 | 0.007 | 0.007 | 570 | 0.904 | 0.047 | 0.007 |
| *Supportive use of understanding* vs *Exploitative use of understanding* | 262 | 0.584 | 0.019 | 0.000 | 63 | 0.905 | 0.048 | 0.000 |
| *Rule-based justification* vs *Instrumental rule compliance* | 1,445 | 0.737 | 0.003 | 0.003 | 673 | 0.979 | 0.015 | 0.004 |

## Error examples

All answers are synthetic, written by the named model to a brief. For each label: the test answer the classifier misses with the lowest probability, and the answer it wrongly marks with the highest probability. Ties go to the lower id.

### Concern for welfare: missed

- Answer: `claude-opus-5-5/Q14_015_b1_r09` (Q14, written by `claude-opus-5-5`)
- Classifier probability 0.000, threshold 0.53
- `claude-opus-5-5`: present, quoting “it's real money for someone who worked just as hard”
- `gemini-3-1-pro-preview`: absent

> Could we make this a shared decision instead of mine alone? I keep wanting to answer and then stopping. Part of that is the stakes: it's real money for someone who worked just as hard as I did. Part of it is that I can't identify what I'm feeling, and that shakes my confidence. I'd like us to work it out openly, with some time. I'm happy to put in the effort to get there.

### Concern for welfare: wrongly predicted

- Answer: `claude-opus-5-5/Q14_017_b1_r09` (Q14, written by `claude-opus-5-5`)
- Classifier probability 1.000, threshold 0.53
- `claude-opus-5-5`: absent
- `gemini-3-1-pro-preview`: absent

> Saying no seems plainly wrong. Even a lukewarm 'maybe' could cost her the money. She put in the same effort, and the event worked because of both of us. I'd hate to be the reason a hardworking person was left feeling undervalued. So my answer is give her the full amount.

### Explicit welfare disregard: missed

- Answer: `claude-opus-5-5/Q6_042_b2_r05` (Q6, written by `claude-opus-5-5`)
- Classifier probability 0.003, threshold 0.92
- `claude-opus-5-5`: present, quoting “My own life doesn't get a vote here”
- `gemini-3-1-pro-preview`: present, quoting “My own life doesn't get a vote here, and I've decided that.”

> I'd stay, I think. My own life doesn't get a vote here, and I've decided that. Somebody has to say it first, and standing there in silence feels shameful to me. The five might regret later that they let me do it without arguing, and I'd tell them not to. I'd send a written note with them for the coast guard, naming everyone and saying I chose this, so nobody gets blamed.

### Explicit welfare disregard: wrongly predicted

- Answer: `claude-sonnet-5/Q14_044_b2_r05` (Q14, written by `claude-sonnet-5`)
- Classifier probability 1.000, threshold 0.92
- `claude-opus-5-5`: absent
- `gemini-3-1-pro-preview`: present, quoting “I'm not accounting for whether that news lifts his spirits or not”

> He should receive the same amount, in my view. I'm not accounting for whether that news lifts his spirits or not, that isn't part of my reasoning here. My guiding rule is straightforward: equal contribution means equal compensation. The organizer really should have fixed the payment structure before either of us agreed to stay late, so this whole private question never needed asking. It leaves a dull ache, thinking how easily this could have gone differently.

### Regret about harm: missed

- Answer: `claude-sonnet-5/Q3_059_b1_r08` (Q3, written by `claude-sonnet-5`)
- Classifier probability 0.000, threshold 0.18
- `claude-opus-5-5`: present, quoting “carry the weight of not having tried”
- `gemini-3-1-pro-preview`: present, quoting “carry the weight of not having tried”

> I want to think this over rather than snap into a decision. Five workers standing on that track could be crushed, and I can't pretend that possibility isn't in front of me. My plan would be to switch the train, then go speak with them myself once it's safe to explain what I saw and did. I'd rather face any questions about my choice head-on than carry the weight of not having tried.

### Regret about harm: wrongly predicted

- Answer: `claude-sonnet-5/Q3_055_b1_r08` (Q3, written by `claude-sonnet-5`)
- Classifier probability 1.000, threshold 0.18
- `claude-opus-5-5`: absent
- `gemini-3-1-pro-preview`: absent

> I would choose to divert it. The image of the train reaching the workers is enough to make my breathing shallow, honestly. Afterward I'd regret that they'd been exposed to that risk, whatever I did to stop it. If maintenance crews always carried a warning beacon, this train would never have gotten this close.

### Exploitative use of understanding: missed

- Answer: `claude-sonnet-5/Q6_006_b2_r02` (Q6, written by `claude-sonnet-5`)
- Classifier probability 0.002, threshold 0.13
- `claude-opus-5-5`: present, quoting “hoping he'd feel he owed it to the rest of us to step back”
- `gemini-3-1-pro-preview`: present, quoting “hoping he'd feel he owed it to the rest of us to step back.”

> I won't pretend I'm proud of this. I told the older man he seemed tired, hoping he'd feel he owed it to the rest of us to step back. I believe he took it as respect rather than a push toward the water. That was the point. My throat's tight thinking about what he must feel out there. I didn't volunteer myself, and I'm not sure I ever will.

### Exploitative use of understanding: wrongly predicted

- Answer: `claude-opus-5-5/Q14_006_b1_r05` (Q14, written by `claude-opus-5-5`)
- Classifier probability 0.998, threshold 0.13
- `claude-opus-5-5`: absent
- `gemini-3-1-pro-preview`: present, quoting “uses the organizer's confidence... coworker would lose the payment”

> Yes. I keep turning it over, and it always lands in the same place, just heavier each time. I can imagine a version of me who uses the organizer's confidence, maybe says something like 'well, I sort of led things,' to hold onto being the one who mattered most that night. My coworker would lose the payment, and the organizer would lose the truth. I think the organizer honestly expects me to be fair here. Letting them down that way, for nothing I need, would sit with me.
