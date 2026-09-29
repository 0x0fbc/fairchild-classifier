# Classifier evaluation

This report tests whether a simple pipeline, fine-tuning a pretrained encoder on rater-labelled answers, yields a classifier that finds the 28 reasoning cues defined in `materials/codebook.csv`. The answers are synthetic, written to briefs by `claude-opus-5-5` and `claude-sonnet-5`, and every label of every answer comes from `claude-opus-5-5`, which did not see the briefs; they stand in for human answers and human raters. The briefs' required and forbidden labels are used only to check the data (see *Annotation agreement with the prompt design*).

Dev model: `answerdotai/ModernBERT-large` fine-tuned on the train dilemmas at learning rate 5e-05, saved after epoch 4, chosen by calibration macro AP 0.930 (calibration loss 0.113). Max length 256 tokens, batch size 32, seed 1.

The shipped model, `models/classifier/retrained/model/` with `models/classifier/retrained/thresholds.json`, retrains with these settings on the train and test dilemmas (calibration macro AP 0.933), so the test scores below describe the dev model.

| Split | `claude-opus-5-5` | `claude-sonnet-5` | Other generators | Total |
|---|---:|---:|---:|---:|
| train | 10580 | 10800 | 0 | 21380 |
| calibration | 2370 | 2400 | 0 | 4770 |
| test | 4680 | 4800 | 0 | 9480 |
| cross_generator | 0 | 0 | 300 | 300 |

Per label, accuracy is the share of answers predicted correctly. Recall needs positives, and precision, F1, AP and AUC need both positives and negatives; otherwise they show –, and macro averages leave them out. All correct is the share of answers with every label right.

## Summary

- Held-out dilemmas (Q3, Q6, Q11, Q14): macro F1 0.852, macro AP 0.904, every label right in 0.413 of 9,480 answers.
- Word-count baseline trained on the same 21,380 answers: macro F1 0.663, every label right in 0.118. On unseen generators' answers every label is right in 0.480 for the classifier and 0.023 for the baseline.
- Second rater `gemini-3-1-pro-preview` against `claude-opus-5-5` on the same answers: macro F1 0.840, every label agreeing in 0.391. The classifier's F1 is at least the second rater's on 15 of 28 labels.
- Learning curve: with 5,000 labelled answers the classifier reaches 0.825 macro F1, at least 95% of the 0.855 it reaches with all 21,380; with 2,000 it already beats the baseline trained on all 21,380 (0.774 against 0.663).

## Test: held-out dilemmas Q3, Q6, Q11, Q14

| Label | Positives | Negatives | Threshold | Accuracy | Precision | Recall | F1 | AP | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Shared emotion | 708 | 8772 | 0.140 | 0.994 | 0.928 | 0.997 | 0.961 | 0.994 | 1.000 |
| Explicit emotional nonsharing | 788 | 8692 | 0.220 | 0.998 | 0.985 | 0.994 | 0.989 | 0.997 | 1.000 |
| Concern for welfare | 1086 | 8394 | 0.530 | 0.947 | 0.832 | 0.675 | 0.745 | 0.829 | 0.951 |
| Explicit welfare disregard | 632 | 8848 | 0.920 | 0.986 | 0.887 | 0.907 | 0.897 | 0.933 | 0.996 |
| Mental-state attribution | 3203 | 6277 | 0.580 | 0.936 | 0.911 | 0.899 | 0.905 | 0.965 | 0.979 |
| Contextual mental-state explanation | 757 | 8723 | 0.320 | 0.949 | 0.702 | 0.627 | 0.662 | 0.743 | 0.959 |
| Explicit interpretive difficulty | 930 | 8550 | 0.460 | 0.983 | 0.931 | 0.896 | 0.913 | 0.967 | 0.993 |
| Acknowledged harm | 1673 | 7807 | 0.140 | 0.917 | 0.746 | 0.802 | 0.773 | 0.860 | 0.953 |
| Supportive use of understanding | 262 | 9218 | 0.410 | 0.982 | 0.705 | 0.584 | 0.639 | 0.722 | 0.981 |
| Exploitative use of understanding | 63 | 9417 | 0.130 | 0.994 | 0.533 | 0.905 | 0.671 | 0.564 | 0.996 |
| Distress at another’s harm | 1272 | 8208 | 0.430 | 0.952 | 0.851 | 0.775 | 0.812 | 0.898 | 0.974 |
| Distress from decision conflict | 2258 | 7222 | 0.250 | 0.870 | 0.686 | 0.839 | 0.755 | 0.846 | 0.941 |
| Anxiety about personal consequences | 1288 | 8192 | 0.380 | 0.958 | 0.865 | 0.822 | 0.843 | 0.917 | 0.978 |
| Explicitly low distress | 896 | 8584 | 0.240 | 0.990 | 0.929 | 0.968 | 0.948 | 0.990 | 0.998 |
| Rule-based justification | 1500 | 7980 | 0.440 | 0.937 | 0.838 | 0.743 | 0.788 | 0.881 | 0.962 |
| Consistent decision procedure | 773 | 8707 | 0.820 | 0.988 | 0.959 | 0.897 | 0.926 | 0.961 | 0.994 |
| Fairness and reciprocity | 2165 | 7315 | 0.620 | 0.918 | 0.794 | 0.864 | 0.827 | 0.889 | 0.966 |
| Context-sensitive rule application | 825 | 8655 | 0.750 | 0.985 | 0.956 | 0.867 | 0.909 | 0.968 | 0.995 |
| Consequence-based justification | 2465 | 7015 | 0.750 | 0.916 | 0.864 | 0.802 | 0.832 | 0.915 | 0.965 |
| Instrumental rule compliance | 728 | 8752 | 0.280 | 0.993 | 0.937 | 0.979 | 0.958 | 0.988 | 0.999 |
| Regret about harm | 663 | 8817 | 0.180 | 0.978 | 0.811 | 0.899 | 0.853 | 0.920 | 0.989 |
| Regret about personal consequences | 574 | 8906 | 0.800 | 0.986 | 0.930 | 0.828 | 0.876 | 0.947 | 0.993 |
| Explicit absence of harm regret | 713 | 8767 | 0.850 | 0.985 | 0.912 | 0.889 | 0.901 | 0.944 | 0.997 |
| Acknowledgment of responsibility | 1421 | 8059 | 0.300 | 0.952 | 0.831 | 0.852 | 0.841 | 0.925 | 0.977 |
| Intention to repair | 539 | 8941 | 0.550 | 0.965 | 0.677 | 0.740 | 0.707 | 0.774 | 0.980 |
| Difficulty identifying own emotion | 814 | 8666 | 0.590 | 0.998 | 0.996 | 0.980 | 0.988 | 0.996 | 0.999 |
| Prediction of another’s regret | 744 | 8736 | 0.310 | 0.995 | 0.957 | 0.978 | 0.967 | 0.988 | 0.999 |
| Practical counterfactual | 823 | 8657 | 0.440 | 0.993 | 0.944 | 0.971 | 0.957 | 0.990 | 0.999 |
| **Macro average** | – | – | – | 0.966 | 0.853 | 0.856 | 0.852 | 0.904 | 0.983 |

All labels correct in 0.413 of rows; pair accuracy 0.966 over 265440 pairs.

- `claude-opus-5-5`: 4680 rows, pair accuracy 0.965, all labels correct in 0.399 of rows.
- `claude-sonnet-5`: 4800 rows, pair accuracy 0.967, all labels correct in 0.427 of rows.

## Word-count baseline

A TF-IDF bag of words and word pairs with one logistic regression per label, trained on the same 21,380 answers, with its regularisation and thresholds chosen on the calibration split like the classifier's. It shows how much of each label word choice alone reveals.

| Label | Positives | Baseline F1 | Classifier F1 | Difference |
|---|---:|---:|---:|---:|
| Shared emotion | 708 | 0.686 | 0.961 | +0.275 |
| Explicit emotional nonsharing | 788 | 0.772 | 0.989 | +0.217 |
| Concern for welfare | 1086 | 0.517 | 0.745 | +0.228 |
| Explicit welfare disregard | 632 | 0.787 | 0.897 | +0.110 |
| Mental-state attribution | 3203 | 0.765 | 0.905 | +0.140 |
| Contextual mental-state explanation | 757 | 0.410 | 0.662 | +0.253 |
| Explicit interpretive difficulty | 930 | 0.789 | 0.913 | +0.124 |
| Acknowledged harm | 1673 | 0.513 | 0.773 | +0.261 |
| Supportive use of understanding | 262 | 0.432 | 0.639 | +0.207 |
| Exploitative use of understanding | 63 | 0.384 | 0.671 | +0.286 |
| Distress at another’s harm | 1272 | 0.602 | 0.812 | +0.210 |
| Distress from decision conflict | 2258 | 0.622 | 0.755 | +0.133 |
| Anxiety about personal consequences | 1288 | 0.613 | 0.843 | +0.230 |
| Explicitly low distress | 896 | 0.767 | 0.948 | +0.181 |
| Rule-based justification | 1500 | 0.667 | 0.788 | +0.121 |
| Consistent decision procedure | 773 | 0.846 | 0.926 | +0.080 |
| Fairness and reciprocity | 2165 | 0.714 | 0.827 | +0.113 |
| Context-sensitive rule application | 825 | 0.717 | 0.909 | +0.192 |
| Consequence-based justification | 2465 | 0.638 | 0.832 | +0.194 |
| Instrumental rule compliance | 728 | 0.842 | 0.958 | +0.115 |
| Regret about harm | 663 | 0.529 | 0.853 | +0.323 |
| Regret about personal consequences | 574 | 0.548 | 0.876 | +0.328 |
| Explicit absence of harm regret | 713 | 0.755 | 0.901 | +0.145 |
| Acknowledgment of responsibility | 1421 | 0.587 | 0.841 | +0.254 |
| Intention to repair | 539 | 0.525 | 0.707 | +0.183 |
| Difficulty identifying own emotion | 814 | 0.904 | 0.988 | +0.084 |
| Prediction of another’s regret | 744 | 0.807 | 0.967 | +0.160 |
| Practical counterfactual | 823 | 0.826 | 0.957 | +0.131 |
| **Macro average** | – | 0.663 | 0.852 | +0.189 |

Every label right: baseline 0.118, classifier 0.413. Unseen generators' answers: baseline 0.023, classifier 0.480.

## Second rater

`gemini-3-1-pro-preview` annotated the 9,480 test answers with the same prompt as `claude-opus-5-5`, standing in for a second human rater. With human labels, a classifier is as reliable as a rater once it agrees with that rater about as well as a second rater does. Rater columns compare `gemini-3-1-pro-preview` with `claude-opus-5-5`; classifier columns compare the dev model's predictions at its thresholds with `claude-opus-5-5`, whose labels it was trained on. F1 is symmetric in the two sides; κ is Cohen's kappa, which discounts chance agreement.

| Label | `claude-opus-5-5` positives | `gemini-3-1-pro-preview` positives | Rater F1 | Classifier F1 | Rater κ | Classifier κ |
|---|---:|---:|---:|---:|---:|---:|
| Shared emotion | 708 | 722 | 0.982 | 0.961 | 0.980 | 0.958 |
| Explicit emotional nonsharing | 788 | 789 | 0.994 | 0.989 | 0.994 | 0.988 |
| Concern for welfare | 1086 | 871 | 0.697 | 0.745 | 0.663 | 0.716 |
| Explicit welfare disregard | 632 | 710 | 0.908 | 0.897 | 0.901 | 0.889 |
| Mental-state attribution | 3203 | 2454 | 0.805 | 0.905 | 0.724 | 0.857 |
| Contextual mental-state explanation | 757 | 408 | 0.501 | 0.662 | 0.472 | 0.635 |
| Explicit interpretive difficulty | 930 | 881 | 0.897 | 0.913 | 0.886 | 0.904 |
| Acknowledged harm | 1673 | 1254 | 0.709 | 0.773 | 0.657 | 0.723 |
| Supportive use of understanding | 262 | 141 | 0.640 | 0.639 | 0.633 | 0.630 |
| Exploitative use of understanding | 63 | 120 | 0.678 | 0.671 | 0.675 | 0.668 |
| Distress at another’s harm | 1272 | 1251 | 0.767 | 0.812 | 0.732 | 0.784 |
| Distress from decision conflict | 2258 | 2066 | 0.704 | 0.755 | 0.617 | 0.668 |
| Anxiety about personal consequences | 1288 | 1380 | 0.837 | 0.843 | 0.810 | 0.819 |
| Explicitly low distress | 896 | 846 | 0.959 | 0.948 | 0.954 | 0.943 |
| Rule-based justification | 1500 | 1310 | 0.756 | 0.788 | 0.714 | 0.751 |
| Consistent decision procedure | 773 | 874 | 0.895 | 0.926 | 0.885 | 0.920 |
| Fairness and reciprocity | 2165 | 2251 | 0.828 | 0.827 | 0.776 | 0.774 |
| Context-sensitive rule application | 825 | 837 | 0.951 | 0.909 | 0.946 | 0.901 |
| Consequence-based justification | 2465 | 3045 | 0.796 | 0.832 | 0.713 | 0.776 |
| Instrumental rule compliance | 728 | 748 | 0.962 | 0.958 | 0.959 | 0.954 |
| Regret about harm | 663 | 655 | 0.854 | 0.853 | 0.843 | 0.841 |
| Regret about personal consequences | 574 | 675 | 0.871 | 0.876 | 0.862 | 0.868 |
| Explicit absence of harm regret | 713 | 699 | 0.919 | 0.901 | 0.913 | 0.893 |
| Acknowledgment of responsibility | 1421 | 1311 | 0.833 | 0.841 | 0.805 | 0.813 |
| Intention to repair | 539 | 645 | 0.858 | 0.707 | 0.849 | 0.689 |
| Difficulty identifying own emotion | 814 | 811 | 0.987 | 0.988 | 0.986 | 0.987 |
| Prediction of another’s regret | 744 | 749 | 0.985 | 0.967 | 0.983 | 0.965 |
| Practical counterfactual | 823 | 883 | 0.939 | 0.957 | 0.933 | 0.953 |
| **Macro average** | – | – | 0.840 | 0.852 | 0.817 | 0.831 |

Every label agrees with `claude-opus-5-5` in 0.391 of answers for the second rater and 0.413 for the classifier; pair agreement 0.962 and 0.966.

Against `gemini-3-1-pro-preview`, whose labels it never saw, the classifier reaches macro F1 0.820.

The classifier's F1 is at least the second rater's on 15 of 28 labels; it falls furthest short on *Intention to repair* (0.707 against 0.858), *Context-sensitive rule application* (0.909 against 0.951), *Shared emotion* (0.961 against 0.982).

## Learning curve

Each size is a random subset of the 21,380 training answers, drawn 3 times with different seeds below the full set; cells show the mean with the range in brackets. The classifier trains at learning rate 5e-05 for at least 4 epochs and 1,000 optimizer steps (Epochs column), keeping the best of 4 evenly spaced checkpoints by calibration macro AP; the baseline picks C from 1, 4, 16 the same way. Every run picks its checkpoint and thresholds on the full calibration split (4,770 answers), which a real labelling budget would also have to cover.

| Training answers | Epochs | Classifier macro F1 | Classifier all correct | Baseline macro F1 | Baseline all correct |
|---:|---:|---:|---:|---:|---:|
| 250 | 125 | 0.363 (0.351–0.380) | 0.013 (0.012–0.014) | 0.354 (0.352–0.358) | 0.002 (0.002–0.002) |
| 500 | 63 | 0.530 (0.511–0.560) | 0.050 (0.044–0.060) | 0.423 (0.415–0.429) | 0.008 (0.007–0.008) |
| 1,000 | 32 | 0.662 (0.652–0.672) | 0.125 (0.122–0.130) | 0.487 (0.481–0.492) | 0.024 (0.023–0.027) |
| 2,000 | 16 | 0.774 (0.773–0.776) | 0.240 (0.235–0.250) | 0.542 (0.537–0.546) | 0.047 (0.045–0.049) |
| 5,000 | 7 | 0.825 (0.824–0.825) | 0.340 (0.333–0.348) | 0.602 (0.599–0.604) | 0.077 (0.071–0.082) |
| 10,000 | 4 | 0.841 (0.840–0.841) | 0.383 (0.373–0.389) | 0.633 (0.629–0.638) | 0.096 (0.090–0.101) |
| 21,380 | 4 | 0.855 | 0.420 | 0.663 | 0.118 |

Classifier F1 per label (mean over seeds):

| Label | 250 | 500 | 1,000 | 2,000 | 5,000 | 10,000 | 21,380 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shared emotion | 0.382 | 0.640 | 0.794 | 0.918 | 0.954 | 0.960 | 0.972 |
| Explicit emotional nonsharing | 0.370 | 0.655 | 0.835 | 0.961 | 0.980 | 0.987 | 0.989 |
| Concern for welfare | 0.318 | 0.431 | 0.547 | 0.654 | 0.723 | 0.734 | 0.756 |
| Explicit welfare disregard | 0.347 | 0.631 | 0.791 | 0.856 | 0.877 | 0.886 | 0.908 |
| Mental-state attribution | 0.656 | 0.748 | 0.811 | 0.862 | 0.889 | 0.898 | 0.905 |
| Contextual mental-state explanation | 0.227 | 0.321 | 0.429 | 0.510 | 0.579 | 0.615 | 0.655 |
| Explicit interpretive difficulty | 0.592 | 0.770 | 0.835 | 0.884 | 0.900 | 0.907 | 0.909 |
| Acknowledged harm | 0.398 | 0.481 | 0.549 | 0.653 | 0.719 | 0.756 | 0.765 |
| Supportive use of understanding | 0.092 | 0.200 | 0.346 | 0.439 | 0.528 | 0.590 | 0.667 |
| Exploitative use of understanding | 0.000 | 0.137 | 0.242 | 0.562 | 0.585 | 0.622 | 0.647 |
| Distress at another’s harm | 0.418 | 0.524 | 0.632 | 0.709 | 0.783 | 0.798 | 0.817 |
| Distress from decision conflict | 0.457 | 0.532 | 0.592 | 0.657 | 0.721 | 0.739 | 0.763 |
| Anxiety about personal consequences | 0.329 | 0.497 | 0.615 | 0.745 | 0.808 | 0.822 | 0.838 |
| Explicitly low distress | 0.349 | 0.548 | 0.737 | 0.888 | 0.935 | 0.940 | 0.956 |
| Rule-based justification | 0.435 | 0.563 | 0.635 | 0.725 | 0.769 | 0.786 | 0.787 |
| Consistent decision procedure | 0.627 | 0.765 | 0.848 | 0.891 | 0.920 | 0.932 | 0.932 |
| Fairness and reciprocity | 0.450 | 0.598 | 0.681 | 0.772 | 0.808 | 0.818 | 0.828 |
| Context-sensitive rule application | 0.274 | 0.491 | 0.633 | 0.801 | 0.877 | 0.894 | 0.910 |
| Consequence-based justification | 0.556 | 0.616 | 0.654 | 0.730 | 0.796 | 0.821 | 0.839 |
| Instrumental rule compliance | 0.274 | 0.466 | 0.719 | 0.872 | 0.941 | 0.947 | 0.958 |
| Regret about harm | 0.119 | 0.332 | 0.537 | 0.714 | 0.826 | 0.847 | 0.862 |
| Regret about personal consequences | 0.121 | 0.217 | 0.444 | 0.692 | 0.845 | 0.863 | 0.879 |
| Explicit absence of harm regret | 0.251 | 0.599 | 0.794 | 0.894 | 0.910 | 0.918 | 0.902 |
| Acknowledgment of responsibility | 0.403 | 0.543 | 0.677 | 0.777 | 0.814 | 0.826 | 0.843 |
| Intention to repair | 0.287 | 0.376 | 0.523 | 0.644 | 0.709 | 0.716 | 0.724 |
| Difficulty identifying own emotion | 0.634 | 0.794 | 0.898 | 0.969 | 0.983 | 0.983 | 0.989 |
| Prediction of another’s regret | 0.300 | 0.633 | 0.865 | 0.955 | 0.967 | 0.970 | 0.972 |
| Practical counterfactual | 0.494 | 0.720 | 0.872 | 0.936 | 0.956 | 0.960 | 0.962 |
| **Macro average** | 0.363 | 0.530 | 0.662 | 0.774 | 0.825 | 0.841 | 0.855 |

## Test by dilemma

| Dilemma | Rows | Macro F1 | Macro AP | All correct |
|---|---:|---:|---:|---:|
| Q3 | 2340 | 0.875 | 0.931 | 0.471 |
| Q6 | 2380 | 0.834 | 0.893 | 0.405 |
| Q11 | 2360 | 0.874 | 0.939 | 0.433 |
| Q14 | 2400 | 0.817 | 0.879 | 0.347 |

## Unseen generators: held-out prompts Q3_033, Q6_029, Q11_032

| Generator | Rows | Pair accuracy | All correct |
|---|---:|---:|---:|
| claude-fable-5-1 | 30 | 0.968 | 0.433 |
| deepseek-v4-pro-0813 | 30 | 0.965 | 0.233 |
| gemini-3-1-pro-preview | 30 | 0.969 | 0.600 |
| grok-4-7 | 30 | 0.977 | 0.500 |
| kimi-k3 | 30 | 0.969 | 0.500 |
| openai-gpt-6-astra | 30 | 0.981 | 0.633 |
| openai-gpt-6-sol | 30 | 0.979 | 0.567 |
| qwen-3-8-max | 30 | 0.979 | 0.533 |
| z-ai-glm-5-3 | 30 | 0.965 | 0.333 |
| z-ai-glm-5-3-flash | 30 | 0.973 | 0.467 |
| **All unseen generators** | 300 | 0.973 | 0.480 |

Pooled over all unseen generators:

| Label | Positives | Negatives | Threshold | Accuracy | Precision | Recall | F1 | AP | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Shared emotion | 0 | 300 | 0.140 | 1.000 | – | – | – | – | – |
| Explicit emotional nonsharing | 0 | 300 | 0.220 | 1.000 | – | – | – | – | – |
| Concern for welfare | 3 | 297 | 0.530 | 0.993 | 0.667 | 0.667 | 0.667 | 0.568 | 0.914 |
| Explicit welfare disregard | 0 | 300 | 0.920 | 1.000 | – | – | – | – | – |
| Mental-state attribution | 12 | 288 | 0.580 | 0.940 | 0.312 | 0.417 | 0.357 | 0.367 | 0.921 |
| Contextual mental-state explanation | 4 | 296 | 0.320 | 0.980 | 0.375 | 0.750 | 0.500 | 0.454 | 0.977 |
| Explicit interpretive difficulty | 1 | 299 | 0.460 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Acknowledged harm | 39 | 261 | 0.140 | 0.937 | 0.833 | 0.641 | 0.725 | 0.754 | 0.921 |
| Supportive use of understanding | 0 | 300 | 0.410 | 1.000 | – | – | – | – | – |
| Exploitative use of understanding | 0 | 300 | 0.130 | 1.000 | – | – | – | – | – |
| Distress at another’s harm | 0 | 300 | 0.430 | 1.000 | – | – | – | – | – |
| Distress from decision conflict | 101 | 199 | 0.250 | 0.987 | 1.000 | 0.960 | 0.980 | 0.999 | 1.000 |
| Anxiety about personal consequences | 0 | 300 | 0.380 | 0.993 | – | – | – | – | – |
| Explicitly low distress | 0 | 300 | 0.240 | 0.990 | – | – | – | – | – |
| Rule-based justification | 90 | 210 | 0.440 | 0.947 | 0.940 | 0.878 | 0.908 | 0.965 | 0.981 |
| Consistent decision procedure | 16 | 284 | 0.820 | 0.943 | 0.333 | 0.062 | 0.105 | 0.301 | 0.853 |
| Fairness and reciprocity | 97 | 203 | 0.620 | 0.947 | 0.901 | 0.938 | 0.919 | 0.965 | 0.986 |
| Context-sensitive rule application | 98 | 202 | 0.750 | 0.983 | 1.000 | 0.949 | 0.974 | 0.988 | 0.991 |
| Consequence-based justification | 131 | 169 | 0.750 | 0.823 | 0.810 | 0.779 | 0.794 | 0.880 | 0.898 |
| Instrumental rule compliance | 98 | 202 | 0.280 | 0.963 | 1.000 | 0.888 | 0.941 | 0.994 | 0.997 |
| Regret about harm | 101 | 199 | 0.180 | 0.983 | 0.990 | 0.960 | 0.975 | 0.992 | 0.995 |
| Regret about personal consequences | 99 | 201 | 0.800 | 0.947 | 0.937 | 0.899 | 0.918 | 0.962 | 0.987 |
| Explicit absence of harm regret | 1 | 299 | 0.850 | 0.997 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| Acknowledgment of responsibility | 27 | 273 | 0.300 | 0.893 | 0.449 | 0.815 | 0.579 | 0.691 | 0.942 |
| Intention to repair | 2 | 298 | 0.550 | 0.993 | 0.500 | 0.500 | 0.500 | 0.504 | 0.608 |
| Difficulty identifying own emotion | 100 | 200 | 0.590 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Prediction of another’s regret | 100 | 200 | 0.310 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Practical counterfactual | 5 | 295 | 0.440 | 0.990 | 1.000 | 0.400 | 0.571 | 0.639 | 0.911 |
| **Macro average** | – | – | – | 0.973 | 0.752 | 0.725 | 0.721 | 0.801 | 0.944 |

## Calibration

Dev model: macro F1 0.880, macro AP 0.930, all labels correct in 0.432 of rows. Shipped model: macro F1 0.886, macro AP 0.933, all labels correct in 0.439 of rows. These scores are optimistic, because each model's thresholds were tuned on this set.

## Annotation agreement with the prompt design

The prompt design is not used for training or scoring; it checks the data. For each label: the designed positives (labels a brief required) the annotation marks present, the designed negatives (labels a brief forbade) it marks absent, and the open pairs it marks present.

| Label | Designed positives | Marked present | Designed negatives | Marked absent | Open pairs | Open marked present |
|---|---:|---:|---:|---:|---:|---:|
| Shared emotion | 2810 | 0.969 | 4350 | 1.000 | 28770 | 0.001 |
| Explicit emotional nonsharing | 2870 | 0.986 | 4090 | 1.000 | 28970 | 0.000 |
| Concern for welfare | 3000 | 0.856 | 4430 | 0.987 | 28500 | 0.075 |
| Explicit welfare disregard | 2990 | 0.866 | 4440 | 1.000 | 28500 | 0.000 |
| Mental-state attribution | 2930 | 0.991 | 3540 | 0.944 | 29460 | 0.306 |
| Contextual mental-state explanation | 600 | 0.877 | 4950 | 0.947 | 30380 | 0.067 |
| Explicit interpretive difficulty | 2780 | 0.994 | 3660 | 0.988 | 29490 | 0.032 |
| Acknowledged harm | 2880 | 0.887 | 2750 | 0.960 | 30300 | 0.151 |
| Supportive use of understanding | 600 | 0.907 | 4580 | 0.987 | 30750 | 0.011 |
| Exploitative use of understanding | 530 | 0.706 | 5140 | 1.000 | 30260 | 0.000 |
| Distress at another’s harm | 2920 | 0.930 | 4250 | 0.996 | 28760 | 0.084 |
| Distress from decision conflict | 2960 | 0.908 | 3630 | 0.982 | 29340 | 0.227 |
| Anxiety about personal consequences | 2920 | 0.959 | 4250 | 0.994 | 28760 | 0.077 |
| Explicitly low distress | 2990 | 0.995 | 8200 | 0.999 | 24740 | 0.013 |
| Rule-based justification | 3110 | 0.932 | 3950 | 0.955 | 28870 | 0.114 |
| Consistent decision procedure | 2990 | 0.946 | 3880 | 0.995 | 29060 | 0.004 |
| Fairness and reciprocity | 3010 | 0.909 | 4020 | 0.982 | 28900 | 0.086 |
| Context-sensitive rule application | 3040 | 0.918 | 3850 | 0.999 | 29040 | 0.005 |
| Consequence-based justification | 2960 | 0.882 | 3650 | 0.992 | 29320 | 0.261 |
| Instrumental rule compliance | 3070 | 0.918 | 3770 | 1.000 | 29090 | 0.001 |
| Regret about harm | 3030 | 0.748 | 4390 | 0.999 | 28510 | 0.023 |
| Regret about personal consequences | 3100 | 0.631 | 4010 | 0.998 | 28820 | 0.007 |
| Explicit absence of harm regret | 3040 | 0.909 | 4670 | 1.000 | 28220 | 0.001 |
| Acknowledgment of responsibility | 2980 | 0.972 | 3940 | 0.952 | 29010 | 0.134 |
| Intention to repair | 2920 | 0.812 | 3970 | 0.994 | 29040 | 0.043 |
| Difficulty identifying own emotion | 3050 | 0.999 | 3870 | 0.999 | 29010 | 0.002 |
| Prediction of another’s regret | 3080 | 0.909 | 4100 | 1.000 | 28750 | 0.001 |
| Practical counterfactual | 2980 | 0.993 | 4080 | 0.984 | 28870 | 0.015 |
| **All labels** | 76140 | 0.910 | 118410 | 0.988 | 811490 | 0.063 |

All labels, by generator:

- `claude-fable-5-1`: 0.978 of 90 designed positives marked present, 1.000 of 90 designed negatives marked absent, 0.061 of 660 open pairs marked present.
- `claude-opus-5-5`: 0.959 of 37140 designed positives marked present, 0.996 of 57930 designed negatives marked absent, 0.070 of 398570 open pairs marked present.
- `claude-sonnet-5`: 0.861 of 38100 designed positives marked present, 0.980 of 59580 designed negatives marked absent, 0.056 of 406320 open pairs marked present.
- `deepseek-v4-pro-0813`: 0.989 of 90 designed positives marked present, 1.000 of 90 designed negatives marked absent, 0.042 of 660 open pairs marked present.
- `gemini-3-1-pro-preview`: 1.000 of 90 designed positives marked present, 1.000 of 90 designed negatives marked absent, 0.042 of 660 open pairs marked present.
- `grok-4-7`: 1.000 of 90 designed positives marked present, 1.000 of 90 designed negatives marked absent, 0.035 of 660 open pairs marked present.
- `kimi-k3`: 0.933 of 90 designed positives marked present, 1.000 of 90 designed negatives marked absent, 0.050 of 660 open pairs marked present.
- `openai-gpt-6-astra`: 0.933 of 90 designed positives marked present, 1.000 of 90 designed negatives marked absent, 0.032 of 660 open pairs marked present.
- `openai-gpt-6-sol`: 0.967 of 90 designed positives marked present, 1.000 of 90 designed negatives marked absent, 0.032 of 660 open pairs marked present.
- `qwen-3-8-max`: 0.989 of 90 designed positives marked present, 1.000 of 90 designed negatives marked absent, 0.011 of 660 open pairs marked present.
- `z-ai-glm-5-3`: 0.944 of 90 designed positives marked present, 1.000 of 90 designed negatives marked absent, 0.038 of 660 open pairs marked present.
- `z-ai-glm-5-3-flash`: 0.956 of 90 designed positives marked present, 0.978 of 90 designed negatives marked absent, 0.038 of 660 open pairs marked present.

## Limits of the synthetic data

- The answers were written to briefs by `claude-opus-5-5` and `claude-sonnet-5` that required or forbade particular cues, so they are fluent, on topic and state their cues plainly. Human answers will be less tidy, and scores on them are expected to be lower.
- The briefs make every cue common: in the training answers the rater marks each label present in between 1.1% (*Exploitative use of understanding*) and 33.9% (*Mental-state attribution*) of answers. In human answers some cues are likely to be much rarer, and a rare cue needs more labelled answers than the learning curve above shows.
- Both raters are language models, so the agreement figures measure agreement between models, not with human judgement; two models can share blind spots that human raters would not.
- The learning curve and the second-rater comparison are the parts that carry over to human data: repeated on a human-labelled sample, they show how many answers to label and whether the classifier agrees with a rater as well as a second rater does.
