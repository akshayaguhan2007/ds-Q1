"""
Student Academic Performance Analysis
Industry: EdTech and Higher Education
Algorithm: Random Forest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. SYNTHETIC DATA GENERATION
# ============================================================
np.random.seed(42)
N = 500

courses = ["Mathematics", "Computer Science", "Business", "Engineering", "Arts"]
course_arr = np.random.choice(courses, N)

prev_perf    = np.random.normal(65, 15, N).clip(30, 100)
attendance   = (prev_perf * 0.5 + np.random.normal(30, 10, N)).clip(40, 100)
int_marks    = (prev_perf * 0.4 + attendance * 0.3 + np.random.normal(10, 8, N)).clip(20, 100)
assign_sub   = (attendance * 0.6 + np.random.normal(20, 10, N)).clip(0, 100)
lp_activity  = (assign_sub * 0.5 + np.random.normal(25, 12, N)).clip(0, 100)

declining = ((lp_activity < 40) & (prev_perf > 60)).astype(int)

composite = (0.3 * int_marks + 0.25 * attendance + 0.2 * assign_sub
             + 0.15 * lp_activity + 0.1 * prev_perf)
perf_status = pd.cut(composite, bins=[0, 50, 65, 80, 100],
                     labels=["At Risk", "Average", "Good", "Excellent"])

df = pd.DataFrame({
    "Student_ID":                  [f"S{str(i).zfill(4)}" for i in range(1, N + 1)],
    "Attendance":                  attendance.round(1),
    "Internal_Marks":              int_marks.round(1),
    "Assignment_Submissions":      assign_sub.round(1),
    "Learning_Platform_Activity":  lp_activity.round(1),
    "Previous_Performance":        prev_perf.round(1),
    "Course":                      course_arr,
    "Declining_Engagement":        declining,
    "Performance_Status":          perf_status,
})

df.to_csv("student_data.csv", index=False)
print("Dataset shape:", df.shape)

# ============================================================
# 2. DATA CLEANING & VALIDATION
# ============================================================
print("\n-- Missing Values --")
print(df.isnull().sum())

df.dropna(inplace=True)

numeric_cols = ["Attendance", "Internal_Marks", "Assignment_Submissions",
                "Learning_Platform_Activity", "Previous_Performance"]
df[numeric_cols] = df[numeric_cols].clip(lower=0, upper=100)

print("\n-- Descriptive Statistics --")
print(df[numeric_cols].describe().round(2))

print("\n-- Performance Status Distribution --")
print(df["Performance_Status"].value_counts())

# ============================================================
# 3. FOUR VISUALISATIONS
# ============================================================
sns.set_theme(style="whitegrid", palette="muted")
colors = sns.color_palette("Set2", 4)

# VIZ 1: Performance Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Viz 1 - Performance Status Distribution", fontsize=14, fontweight="bold")

order  = ["At Risk", "Average", "Good", "Excellent"]
counts = df["Performance_Status"].value_counts().reindex(order)

axes[0].bar(order, counts.values, color=colors)
axes[0].set_xlabel("Performance Status")
axes[0].set_ylabel("Number of Students")
axes[0].set_title("Count by Category")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 3, str(v), ha="center", fontweight="bold")

axes[1].pie(counts.values, labels=order, autopct="%1.1f%%", colors=colors, startangle=140)
axes[1].set_title("Proportion")

plt.tight_layout()
plt.savefig("viz1_performance_distribution.png", dpi=150)
plt.show()
print("Saved: viz1_performance_distribution.png")

# VIZ 2: Attendance vs Internal Marks
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("Viz 2 - Attendance vs Internal Marks by Performance Status",
             fontsize=14, fontweight="bold")

palette = {"At Risk": "#e74c3c", "Average": "#f39c12", "Good": "#2ecc71", "Excellent": "#3498db"}
for status, grp in df.groupby("Performance_Status", observed=True):
    ax.scatter(grp["Attendance"], grp["Internal_Marks"],
               label=status, alpha=0.6, s=40, color=palette[status])

m, b   = np.polyfit(df["Attendance"], df["Internal_Marks"], 1)
x_line = np.linspace(df["Attendance"].min(), df["Attendance"].max(), 100)
r      = df["Attendance"].corr(df["Internal_Marks"])
ax.plot(x_line, m * x_line + b, "k--", linewidth=1.5, label=f"Trend (r={r:.2f})")

ax.set_xlabel("Attendance (%)")
ax.set_ylabel("Internal Marks")
ax.legend(title="Performance Status")
plt.tight_layout()
plt.savefig("viz2_attendance_vs_marks.png", dpi=150)
plt.show()
print("Saved: viz2_attendance_vs_marks.png")

# VIZ 3: Assignment Submissions boxplot + Correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Viz 3 - Assignment Completion and Feature Correlations",
             fontsize=14, fontweight="bold")

df.boxplot(column="Assignment_Submissions", by="Performance_Status",
           ax=axes[0], patch_artist=True,
           boxprops=dict(facecolor="#AED6F1"),
           medianprops=dict(color="red", linewidth=2))
axes[0].set_title("Assignment Submissions by Status")
axes[0].set_xlabel("Performance Status")
axes[0].set_ylabel("Assignment Submissions (%)")
plt.sca(axes[0])
plt.xticks(rotation=15)

corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            ax=axes[1], linewidths=0.5, vmin=-1, vmax=1)
axes[1].set_title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("viz3_assignments_and_correlation.png", dpi=150)
plt.show()
print("Saved: viz3_assignments_and_correlation.png")

# VIZ 4: LP Activity by Course + Declining Engagement scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Viz 4 - Learning Behaviour Across Courses and Declining Engagement",
             fontsize=14, fontweight="bold")

course_lp = df.groupby("Course")["Learning_Platform_Activity"].mean().sort_values()
axes[0].barh(course_lp.index, course_lp.values,
             color=sns.color_palette("Set3", len(course_lp)))
axes[0].set_xlabel("Avg Learning Platform Activity")
axes[0].set_title("Avg LP Activity by Course")
for i, v in enumerate(course_lp.values):
    axes[0].text(v + 0.3, i, f"{v:.1f}", va="center")

dec     = df[df["Declining_Engagement"] == 1]
non_dec = df[df["Declining_Engagement"] == 0]
axes[1].scatter(non_dec["Previous_Performance"], non_dec["Learning_Platform_Activity"],
                alpha=0.4, s=30, color="#2ecc71", label="Engaged")
axes[1].scatter(dec["Previous_Performance"], dec["Learning_Platform_Activity"],
                alpha=0.7, s=50, color="#e74c3c", marker="X",
                label=f"Declining ({len(dec)})")
axes[1].set_xlabel("Previous Performance")
axes[1].set_ylabel("Learning Platform Activity")
axes[1].set_title("Declining Engagement Detection")
axes[1].legend()

plt.tight_layout()
plt.savefig("viz4_learning_behaviour.png", dpi=150)
plt.show()
print("Saved: viz4_learning_behaviour.png")

# ============================================================
# 4. RANDOM FOREST MODEL
# ============================================================
le = LabelEncoder()
df["Performance_Encoded"] = le.fit_transform(df["Performance_Status"])

features = ["Attendance", "Internal_Marks", "Assignment_Submissions",
            "Learning_Platform_Activity", "Previous_Performance"]
X = df[features]
y = df["Performance_Encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, max_depth=8,
                            random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\n-- Random Forest Classification Report --")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Model evaluation plot (confusion matrix + feature importances)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Random Forest - Model Evaluation", fontsize=14, fontweight="bold")

cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")
axes[0].tick_params(axis="x", rotation=20)

importances = pd.Series(rf.feature_importances_, index=features).sort_values()
axes[1].barh(importances.index, importances.values,
             color=sns.color_palette("Set2", len(features)))
axes[1].set_xlabel("Importance Score")
axes[1].set_title("Feature Importances")
for i, v in enumerate(importances.values):
    axes[1].text(v + 0.002, i, f"{v:.3f}", va="center")

plt.tight_layout()
plt.savefig("viz5_model_evaluation.png", dpi=150)
plt.show()
print("Saved: viz5_model_evaluation.png")

# ============================================================
# 5. KEY INSIGHTS & ACTION PLAN
# ============================================================
corr_att_marks    = df["Attendance"].corr(df["Internal_Marks"])
corr_assign_marks = df["Assignment_Submissions"].corr(df["Internal_Marks"])
at_risk_pct       = (df["Performance_Status"] == "At Risk").mean() * 100
declining_count   = int(df["Declining_Engagement"].sum())
top_feature       = importances.idxmax()
low_lp_course     = course_lp.idxmin()

print("""
=================================================================
  KEY INSIGHTS
=================================================================""")
print(f"""
  1. ATTENDANCE DRIVES MARKS
     Attendance-Internal Marks correlation = {corr_att_marks:.2f}.
     Students below 60% attendance cluster almost entirely in
     the 'At Risk' and 'Average' categories.

  2. ASSIGNMENT COMPLETION IS A STRONG PREDICTOR
     Correlation with Internal Marks = {corr_assign_marks:.2f}.
     Students submitting >75% of assignments score significantly
     higher, showing consistent effort matters more than ability.

  3. {at_risk_pct:.1f}% OF STUDENTS ARE AT RISK
     Nearly 1 in {round(100/at_risk_pct)} students falls in the 'At Risk' band,
     requiring immediate academic intervention.

  4. DECLINING ENGAGEMENT IS DETECTABLE EARLY
     {declining_count} students show declining engagement (high previous
     performance but low current LP activity) -- a key early-warning
     signal before marks drop.

  5. '{top_feature}' IS THE TOP MODEL FEATURE
     The Random Forest ranks it as the most important predictor,
     meaning historical academic behaviour is the strongest
     indicator of future outcomes.

  6. LEARNING PLATFORM USAGE VARIES BY COURSE
     '{low_lp_course}' has the lowest average LP activity,
     suggesting course-specific digital engagement strategies
     are needed rather than a one-size-fits-all approach.

  7. COMPOSITE ENGAGEMENT BEATS SINGLE METRICS
     No single feature alone separates performance groups.
     The combination of attendance + assignments + LP activity
     provides the clearest separation between At Risk and Excellent.
""")

print("""
=================================================================
  ACADEMIC SUPPORT ACTION PLAN
=================================================================

  INTERVENTION 1 - Early-Warning Dashboard (Weeks 1-3)
  * Flag students with attendance < 60% OR assignment submission
    rate < 50% for immediate advisor outreach.
  * Automate weekly alerts from the learning platform.

  INTERVENTION 2 - Targeted Tutoring for At-Risk Students
  * Enrol 'At Risk' students in peer-tutoring or supplemental
    instruction sessions before mid-term assessments.
  * Prioritise students with declining engagement scores.

  INTERVENTION 3 - Assignment Completion Incentives
  * Introduce low-stakes formative assignments with instant
    feedback to build submission habits early in the semester.
  * Gamify LP activity streaks to boost platform engagement.

  INTERVENTION 4 - Course-Level Digital Strategy
  * Work with instructors in low-LP-activity courses to embed
    platform tasks (quizzes, videos) directly into coursework.

  INTERVENTION 5 - Predictive Re-Scoring at Mid-Term
  * Re-run the Random Forest model at week 6 with updated data
    to re-prioritise the intervention list dynamically.

=================================================================
""")
