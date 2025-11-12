# MCC Computation Fix: Prefix-Grouped Macro-Averaging

## Problem

The original MCC computation used simple micro-averaging across all binary concepts:

```python
# OLD: Micro-average (treats all concepts equally)
top_scores = scores.max(dim=0).values
mcc = top_scores.mean().item()  # Simple mean
```

**Issue**: This gives equal weight to each binary concept, which is problematic when some groups have more classes than others.

Example:
- `domain`: 4 classes (fantasy, science, news, other)
- `sentiment`: 3 classes (positive, neutral, negative)
- `voice`: 2 classes (active, passive)
- `tense`: 2 classes (present, past)

Simple averaging would weight `domain` 4x more than `tense`!

---

## Solution: Prefix-Grouped Macro-Averaging

The fixed version groups concepts by prefix and computes a macro-average:

```python
# NEW: Macro-average (groups by concept prefix)
# 1. Group by prefix
prefix_groups = {}
for i, label in enumerate(list(top_features.keys())):
    prefix = label.split('-')[0]  # "domain-science" → "domain"
    if prefix not in prefix_groups:
        prefix_groups[prefix] = []
    prefix_groups[prefix].append(top_scores[i].item())

# 2. Average within each group
prefix_averages = {}
for prefix, group_scores in prefix_groups.items():
    prefix_averages[prefix] = sum(group_scores) / len(group_scores)

# 3. Macro-average across groups
mcc = sum(prefix_averages.values()) / len(prefix_averages)
```

---

## Example

### Input Scores:
```
domain-fantasy:  0.850
domain-science:  0.820
domain-news:     0.790
domain-other:    0.760
sentiment-positive: 0.900
sentiment-neutral:  0.880
sentiment-negative: 0.870
voice-active:    0.950
voice-passive:   0.940
tense-present:   0.800
tense-past:      0.780
```

### Step 1: Group by Prefix
```
domain:    [0.850, 0.820, 0.790, 0.760]
sentiment: [0.900, 0.880, 0.870]
voice:     [0.950, 0.940]
tense:     [0.800, 0.780]
```

### Step 2: Compute Prefix Averages
```
domain:    (0.850 + 0.820 + 0.790 + 0.760) / 4 = 0.805
sentiment: (0.900 + 0.880 + 0.870) / 3 = 0.883
voice:     (0.950 + 0.940) / 2 = 0.945
tense:     (0.800 + 0.780) / 2 = 0.790
```

### Step 3: Macro-Average Across Groups
```
MCC = (0.805 + 0.883 + 0.945 + 0.790) / 4 = 0.856
```

### Comparison with Old Method:
```
# OLD (micro-average):
MCC = (0.850 + 0.820 + 0.790 + 0.760 + 0.900 + 0.880 + 0.870 + 0.950 + 0.940 + 0.800 + 0.780) / 11
    = 9.340 / 11
    = 0.849

# NEW (macro-average):
MCC = 0.856
```

Small difference in this example, but becomes significant with imbalanced groups.

---

## Output Format

### Before:
```
MCC:
Max Correlation (MCC) for each concept:
--------------------------------------------------
  domain-fantasy: 0.8500 (feature 1234)
  domain-science: 0.8200 (feature 2345)
  ...
  tense-past: 0.7800 (feature 9012)

Average MCC: 0.8491
```

### After:
```
MCC:
Max Correlation (MCC) for each concept:
--------------------------------------------------
  domain-fantasy: 0.8500 (feature 1234)
  domain-science: 0.8200 (feature 2345)
  ...
  tense-past: 0.7800 (feature 9012)

Prefix Group Averages:
--------------------------------------------------
  domain: 0.8050
  sentiment: 0.8833
  tense: 0.7900
  voice: 0.9450

Macro-Average MCC (across 4 groups): 0.8558
```

---

## Why This Matters

### Benefits:
1. ✅ **Fair weighting**: Each concept group (domain, sentiment, etc.) gets equal weight
2. ✅ **Interpretable**: Can see performance per concept group
3. ✅ **Standard practice**: Macro-averaging is standard for imbalanced multi-class problems
4. ✅ **Comparable**: Results are more comparable across different concept sets

### Use Cases:
- Comparing models trained on different concept subsets
- Understanding which concept groups are easier/harder to probe
- Fair evaluation when concept groups have different numbers of classes

---

## Code Location

**File**: `scripts/train_probes.py`
**Lines**: 3910-3943
**Function**: Output section of `main()` when `metric == "mcc"`

---

## Related Functions

- `score_identification()`: Computes correlation matrix (lines 341-523)
- `evaluate_sentence_labels()`: Main evaluation function (lines 3240-3382)
- `sweep_k_values_for_plots()`: K-sweep uses similar concept grouping (lines 1575-1659)

---

## Note on Terminology

**Micro-average**: Average across all instances (treats each binary concept equally)
**Macro-average**: Average across groups (treats each concept group equally)

For imbalanced groups, macro-average is generally preferred to avoid bias toward groups with more classes.
