# ds-Q1 — Student Academic Performance Analysis

## Overview
EdTech / Higher Education data science project that analyses factors associated with student underperformance and trains a Random Forest classifier to predict performance categories.

## Dataset
500 synthetic students with the following features:
- `Attendance` — class attendance percentage
- `Internal_Marks` — internal assessment score
- `Assignment_Submissions` — assignment completion rate
- `Learning_Platform_Activity` — online platform engagement score
- `Previous_Performance` — prior academic score
- `Course` — enrolled course (Mathematics, Computer Science, Business, Engineering, Arts)
- `Performance_Status` — target label (At Risk / Average / Good / Excellent)

## Project Structure
```
student_performance.py   # Main analysis script
student_data.csv         # Generated dataset
viz1_performance_distribution.png
viz2_attendance_vs_marks.png
viz3_assignments_and_correlation.png
viz4_learning_behaviour.png
viz5_model_evaluation.png
```

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python student_performance.py
```

## Key Results
| Metric | Value |
|---|---|
| Model | Random Forest (100 trees, max_depth=8) |
| Overall Accuracy | 86% |
| Top Predictive Feature | Attendance |
| At-Risk Students | ~16% of cohort |
| Declining Engagement Detected | 32 students |

## 7 Key Insights
1. Attendance is the strongest single predictor of marks (r = 0.59)
2. Assignment completion correlates with marks (r = 0.36)
3. 16% of students are At Risk and need immediate intervention
4. 32 students show declining engagement detectable before marks drop
5. Attendance is the top Random Forest feature
6. Engineering has the lowest learning platform activity
7. Composite engagement (attendance + assignments + LP) separates groups best

## Action Plan
1. Early-warning dashboard with automated alerts (weeks 1-3)
2. Targeted tutoring for At Risk students before mid-terms
3. Gamified assignment completion incentives
4. Course-level digital engagement strategy
5. Re-run model at week 6 to dynamically update intervention list
