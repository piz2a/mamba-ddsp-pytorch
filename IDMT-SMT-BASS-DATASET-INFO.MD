# IDMT-SMT-BASS Dataset Inventory

Generated: 2026-07-13

## Scope

This report covers:

- `/disk1/ahnjiho/IDMT-SMT-BASS`
- `/disk1/ahnjiho/IDMT-SMT-BASS-SINGLE-TRACKS`

Label handling:

- PS means plucking / excitation style.
- ES means expression style.
- ES subclasses are collapsed into their main ES class for the main tables:
  - `BEQ`, `BES` -> `BE`
  - `SLD`, `SLU` -> `SL`
  - `VIF`, `VIS` -> `VI`
- A separate table shows the stricter training-filter case where raw ES subclass files are removed.

Counting method:

- `IDMT-SMT-BASS`: one sample is one isolated-note WAV file. Labels were parsed from the WAV filename.
- `IDMT-SMT-BASS-SINGLE-TRACKS`: one sample is one annotated XML `<event>` note. Labels were parsed from `annotation/*.xml`.

## High-Level Counts

| Dataset | Audio files | Annotation files used | Count unit | Raw count | Deduplicated count |
|---|---:|---:|---|---:|---:|
| IDMT-SMT-BASS | 5,306 WAV | filename labels | isolated WAV | 5,306 | 4,837 |
| IDMT-SMT-BASS-SINGLE-TRACKS | 17 WAV | 17 XML | annotated note event | 948 | 948 |

Important: `IDMT-SMT-BASS` contains 469 duplicated logical WAV filenames. These are the same `FS_NO` isolated notes appearing in both `PS/FS` and `ES/NO`. For training statistics, the deduplicated count of 4,837 is the more useful number.

## IDMT-SMT-BASS: Deduplicated Main ES x PS Counts

This is the recommended table for estimating independent isolated-note coverage.

| ES \ PS | FS | MU | PK | SP | ST | Total |
|---|---:|---:|---:|---:|---:|---:|
| NO | 469 | 468 | 468 | 469 | 468 | 2,342 |
| BE | 468 | 0 | 0 | 0 | 0 | 468 |
| DN | 468 | 0 | 0 | 0 | 0 | 468 |
| HA | 467 | 0 | 0 | 0 | 0 | 467 |
| SL | 312 | 0 | 0 | 0 | 0 | 312 |
| VI | 780 | 0 | 0 | 0 | 0 | 780 |
| Total | 2,964 | 468 | 468 | 469 | 468 | 4,837 |

### IDMT-SMT-BASS: Physical WAV Counts

This table counts every WAV path recursively, including the 469 duplicate `FS_NO` files.

| ES \ PS | FS | MU | PK | SP | ST | Total |
|---|---:|---:|---:|---:|---:|---:|
| NO | 938 | 468 | 468 | 469 | 468 | 2,811 |
| BE | 468 | 0 | 0 | 0 | 0 | 468 |
| DN | 468 | 0 | 0 | 0 | 0 | 468 |
| HA | 467 | 0 | 0 | 0 | 0 | 467 |
| SL | 312 | 0 | 0 | 0 | 0 | 312 |
| VI | 780 | 0 | 0 | 0 | 0 | 780 |
| Total | 3,433 | 468 | 468 | 469 | 468 | 5,306 |

### IDMT-SMT-BASS: Raw ES Counts Before Subclass Collapse

Deduplicated by filename:

| Raw ES | Count |
|---|---:|
| NO | 2,342 |
| BE | 156 |
| BEQ | 156 |
| BES | 156 |
| DN | 468 |
| HA | 467 |
| SLD | 156 |
| SLU | 156 |
| VI | 468 |
| VIF | 156 |
| VIS | 156 |
| Total | 4,837 |

### IDMT-SMT-BASS: If Raw ES Subclasses Are Removed

This is the stricter training subset if `BEQ`, `BES`, `SLD`, `SLU`, `VIF`, and `VIS` are removed instead of collapsed.

Deduplicated by filename:

| ES \ PS | FS | MU | PK | SP | ST | Total |
|---|---:|---:|---:|---:|---:|---:|
| NO | 469 | 468 | 468 | 469 | 468 | 2,342 |
| BE | 156 | 0 | 0 | 0 | 0 | 156 |
| DN | 468 | 0 | 0 | 0 | 0 | 468 |
| HA | 467 | 0 | 0 | 0 | 0 | 467 |
| SL | 0 | 0 | 0 | 0 | 0 | 0 |
| VI | 468 | 0 | 0 | 0 | 0 | 468 |
| Total | 2,028 | 468 | 468 | 469 | 468 | 3,901 |

Physical WAV paths after this filter: 4,370.

## IDMT-SMT-BASS-SINGLE-TRACKS: Main ES x PS Event Counts

| ES \ PS | FS | MU | PK | SP | ST | Total |
|---|---:|---:|---:|---:|---:|---:|
| NO | 346 | 191 | 144 | 46 | 102 | 829 |
| BE | 2 | 6 | 0 | 0 | 0 | 8 |
| DN | 0 | 6 | 12 | 0 | 10 | 28 |
| HA | 32 | 0 | 0 | 0 | 0 | 32 |
| SL | 12 | 2 | 2 | 4 | 0 | 20 |
| VI | 16 | 9 | 4 | 0 | 2 | 31 |
| Total | 408 | 214 | 162 | 50 | 114 | 948 |

Raw ES labels in the single-track XML are already main classes:

| ES | Count |
|---|---:|
| NO | 829 |
| BE | 8 |
| DN | 28 |
| HA | 32 |
| SL | 20 |
| VI | 31 |
| Total | 948 |

## Marginal Distributions

### IDMT-SMT-BASS, Deduplicated

| ES | Count | Percent |
|---|---:|---:|
| NO | 2,342 | 48.42% |
| BE | 468 | 9.68% |
| DN | 468 | 9.68% |
| HA | 467 | 9.66% |
| SL | 312 | 6.45% |
| VI | 780 | 16.13% |
| Total | 4,837 | 100.00% |

| PS | Count | Percent |
|---|---:|---:|
| FS | 2,964 | 61.28% |
| MU | 468 | 9.68% |
| PK | 468 | 9.68% |
| SP | 469 | 9.70% |
| ST | 468 | 9.68% |
| Total | 4,837 | 100.00% |

### IDMT-SMT-BASS-SINGLE-TRACKS

| ES | Count | Percent |
|---|---:|---:|
| NO | 829 | 87.45% |
| BE | 8 | 0.84% |
| DN | 28 | 2.95% |
| HA | 32 | 3.38% |
| SL | 20 | 2.11% |
| VI | 31 | 3.27% |
| Total | 948 | 100.00% |

| PS | Count | Percent |
|---|---:|---:|
| FS | 408 | 43.04% |
| MU | 214 | 22.57% |
| PK | 162 | 17.09% |
| SP | 50 | 5.27% |
| ST | 114 | 12.03% |
| Total | 948 | 100.00% |

## Duration and Coverage Statistics

### IDMT-SMT-BASS

Using deduplicated isolated WAVs:

| Metric | Value |
|---|---:|
| Sample rate | 44,100 Hz |
| Unique isolated-note WAVs | 4,837 |
| Total audio duration | 11,892.54 s / 198.21 min / 3.30 h |
| Mean WAV duration | 2.459 s |
| Median WAV duration | 2.448 s |
| 10th percentile duration | 1.681 s |
| 90th percentile duration | 3.241 s |
| Minimum duration | 0.863 s |
| Maximum duration | 23.769 s |

Physical WAV paths including duplicates:

| Metric | Value |
|---|---:|
| Physical WAV paths | 5,306 |
| Total audio duration | 13,359.67 s / 222.66 min / 3.71 h |
| Mean WAV duration | 2.518 s |
| Median WAV duration | 2.500 s |

Other numeric traits:

- String counts after deduplication: string 1 = 1,209, string 2 = 1,209, string 3 = 1,209, string 4 = 1,208, string 5 = 2.
- The string 5 entries are outliers for a 4-string bass dataset and should be inspected or filtered.
- Fret range after deduplication: 0 to 14, 15 unique fret values.
- The longest file is `PS/MU/BS_3_EQ_3_MU_NO_4_12.wav` at about 23.77 s. This is a strong duration outlier.

### IDMT-SMT-BASS-SINGLE-TRACKS

| Metric | Value |
|---|---:|
| Track WAV files | 17 |
| XML annotation files | 17 |
| Annotated note events | 948 |
| Total track audio duration | 351.97 s / 5.87 min |
| Total annotated note duration | 281.68 s / 4.69 min |
| Annotated-note occupancy | 80.03% |
| Rest / non-event duration | 70.29 s |
| Mean track duration | 20.704 s |
| Median track duration | 20.000 s |
| Minimum track duration | 14.769 s |
| Maximum track duration | 29.538 s |
| Mean annotated note duration | 0.297 s |
| Median annotated note duration | 0.253 s |
| 10th percentile note duration | 0.150 s |
| 90th percentile note duration | 0.490 s |
| Minimum note duration | 0.064 s |
| Maximum note duration | 0.915 s |

Other numeric traits:

- String counts: string 1 = 230, string 2 = 326, string 3 = 260, string 4 = 132.
- Fret range: 0 to 12, 11 unique fret values.
- The single-track set is much more realistic rhythmically, but much smaller and heavily dominated by `NO`.

## Statistical Interpretation

1. `IDMT-SMT-BASS` is much larger in isolated-note coverage, but its class structure is artificial.
   The main non-`NO` expression classes are almost exclusively `FS`. `MU`, `PK`, `SP`, and `ST` appear almost only with `NO`. This means it is weak evidence for learning independent PS x ES combinations.

2. `IDMT-SMT-BASS-SINGLE-TRACKS` has real phrase context, note boundaries, rests, and short rhythmic durations.
   It is better for learning gate, onset, offset, note-age, note-progress, and transition behavior. However, it only has 948 annotated note events.

3. The single-track set is highly imbalanced toward `NO`.
   `NO` is 87.45% of events. All expressive categories are sparse: `BE` has only 8 events, `SL` 20, `DN` 28, `VI` 31, and `HA` 32.

4. The isolated-note set is also imbalanced after deduplication.
   `FS` is 61.28% of isolated notes, and `NO` is 48.42% of ES labels. This is not fatal, but it means class balancing or sampling weights are needed.

5. Removing F0-dependent ES subclasses from `IDMT-SMT-BASS` reduces the deduplicated isolated-note set from 4,837 to 3,901.
   It also removes all `SL` coverage because isolated-note `SL` appears only as `SLD` and `SLU`.

6. For Bass-DDSP v2, the best practical use is:
   - Use deduplicated `IDMT-SMT-BASS` for isolated-note timbre, transient, and articulation reconstruction.
   - Use `IDMT-SMT-BASS-SINGLE-TRACKS` for sequence/gate/onset/offset behavior and realistic duration distributions.
   - Do not assume full PS and ES independence from these datasets. Prefer observed articulation classes such as `FS_NO`, `PK_NO`, `FS_HA`, etc.

## Recommended Training Implications

- Deduplicate `IDMT-SMT-BASS` by filename before building train/validation splits.
- Filter or inspect string 5 entries in `IDMT-SMT-BASS`.
- Filter or cap extreme duration outliers before random riff generation.
- If the goal is to remove F0-dependent ES subclasses, use the 3,901-sample deduplicated subset.
- If the goal is broad main-class coverage, collapse subclasses and keep the 4,837-sample deduplicated set.
- For observed-articulation modeling, do not synthesize unsupported PS x ES pairs as if they were observed. In the isolated-note set, most expressive ES classes are only observed with `FS`.
- Use the single-track set as a sequence-conditioning supplement, not as the main timbre dataset.
