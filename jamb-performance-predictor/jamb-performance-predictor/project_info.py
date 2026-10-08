"""Shared constants: thesis results, feature labels and project facts."""

TITLE = "Predicting Student's Academic Performance Using Machine Learning Models"
AUTHOR = "Olesin Ibukunoluwa Moyosore"
SUPERVISOR = "Dr. Odegbesan Omobolaji"
INSTITUTION = "Department of Computer Science, Caleb University, Imota, Lagos"
YEAR = "2025"
DATASET_URL = "https://www.kaggle.com/datasets/idowuadamo/students-performance-in-2024-jamb"
GITHUB_PROFILE = "https://github.com/Moyohor"
PASS_MARK = 200

# Table 4.1 of the thesis (test set, 80/20 split, random_state=42, target = raw JAMB score)
THESIS_METRICS = [
    {"Model": "Logistic Regression",    "RMSE": 39.7529, "MAE": 31.8862, "R2": 0.3452},
    {"Model": "K-Nearest Neighbors",    "RMSE": 44.0931, "MAE": 35.6460, "R2": 0.1944},
    {"Model": "Random Forest",          "RMSE": 40.6270, "MAE": 32.9669, "R2": 0.3161},
    {"Model": "Support Vector Machine", "RMSE": 47.7026, "MAE": 37.1240, "R2": 0.0571},
]

# feature -> (label, help text adapted from Table 3.1 of the thesis)
FEATURES = {
    "Study_Hours_Per_Week": ("Study hours per week", "Average hours spent studying each week outside class."),
    "Attendance_Rate": ("Attendance rate (%)", "Share of classes attended over the period."),
    "Assignments_Completed": ("Assignments completed", "Assignments submitted within the given timeframe."),
    "Teacher_Quality": ("Teacher quality", "Effectiveness of teaching, on the dataset's scale."),
    "Distance_To_School": ("Distance to school (km)", "Distance from home to school; can affect punctuality and fatigue."),
    "School_Type": ("School type", "Public or private school."),
    "School_Location": ("School location", "Urban or rural setting."),
    "Extra_Tutorials": ("Extra tutorials", "Whether the student attends additional tutoring."),
    "Access_To_Learning_Materials": ("Access to learning materials", "Availability of textbooks, internet, library or digital resources."),
    "Parent_Involvement": ("Parental involvement", "How actively parents or guardians take part in schooling."),
    "IT_Knowledge": ("IT knowledge", "Familiarity with information technology."),
    "Age": ("Age", "Age in years."),
    "Gender": ("Gender", "Gender as recorded in the dataset."),
    "Socioeconomic_Status": ("Socioeconomic status", "Family income level, living conditions and access to resources."),
    "Parent_Education_Level": ("Parent education level", "Highest qualification attained by the parents."),
}

GROUPS = {
    "Study habits": ["Study_Hours_Per_Week", "Attendance_Rate", "Assignments_Completed", "Extra_Tutorials"],
    "School": ["School_Type", "School_Location", "Distance_To_School", "Teacher_Quality", "Access_To_Learning_Materials"],
    "Home and background": ["Parent_Involvement", "Parent_Education_Level", "Socioeconomic_Status",
                            "IT_Knowledge", "Age", "Gender"],
}

# Sample profiles for the quick-fill buttons
STRONG_SAMPLE = {
    "Study_Hours_Per_Week": 20, "Attendance_Rate": 85, "Teacher_Quality": 3, "Distance_To_School": 5.0,
    "School_Type": "Private", "School_Location": "Urban", "Extra_Tutorials": "Yes",
    "Access_To_Learning_Materials": "Yes", "Parent_Involvement": "High", "IT_Knowledge": "High",
    "Age": 18, "Gender": "Female", "Socioeconomic_Status": "Medium",
    "Parent_Education_Level": "Tertiary", "Assignments_Completed": 2,
}
AT_RISK_SAMPLE = {
    "Study_Hours_Per_Week": 5, "Attendance_Rate": 23, "Teacher_Quality": 2, "Distance_To_School": 5.08,
    "School_Type": "Public", "School_Location": "Rural", "Extra_Tutorials": "No",
    "Access_To_Learning_Materials": "No", "Parent_Involvement": "Low", "IT_Knowledge": "Low",
    "Age": 18, "Gender": "Male", "Socioeconomic_Status": "Low",
    "Parent_Education_Level": "Primary", "Assignments_Completed": 0,
}
