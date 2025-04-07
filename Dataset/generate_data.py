import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 500

# Generate synthetic features
student_ids = np.arange(1, num_samples + 1)
study_hours = np.round(np.random.uniform(1, 10, num_samples), 1)  # Study hours per day (1-10 hours)
attendance = np.round(np.random.uniform(50, 100, num_samples), 1)  # Attendance percentage (50-100%)
assignment_score = np.round(np.random.uniform(40, 100, num_samples), 1)  # Average assignment score (40-100)
last_sem_percentage = np.round(np.random.uniform(40, 100, num_samples), 1)  # Last semester percentage (40-100)
mobile_screen_time = np.round(np.random.uniform(1, 6, num_samples), 1)  # Mobile screen time in hours per day (1-6)
sleep_hours = np.round(np.random.uniform(4, 10, num_samples), 1)  # Average sleep hours per day (4-10)

# Compute a synthetic "Final_Score" as a weighted sum of some features plus noise
# Weights are chosen arbitrarily to simulate influence:
# - Last semester percentage contributes 30%
# - Assignment score contributes 40%
# - Study hours (scaled) contributes 20%
# - Attendance contributes 10%
# Plus some random noise
final_score = (
    0.3 * last_sem_percentage +
    0.4 * assignment_score +
    0.2 * (study_hours * 10) +  # scaling study hours to be comparable
    0.1 * attendance 
)
# Add random noise
noise = np.random.normal(0, 5, num_samples)
final_score = final_score + noise
final_score = np.clip(final_score, 0, 100)  # ensure scores are within 0 to 100
final_score = np.round(final_score, 1)

# Determine Pass/Fail based on a standard rule (passing if final score >= 40)
pass_fail = np.where(final_score >= 40, 1, 0)

# Create a DataFrame with all the features
df = pd.DataFrame({
    'Student_ID': student_ids,
    'Study_Hours': study_hours,
    'Attendance (%)': attendance,
    'Assignment_Score': assignment_score,
    'Last_Sem_Percentage': last_sem_percentage,
    'Mobile_Screen_Time': mobile_screen_time,
    'Sleep_Hours': sleep_hours,
    'Final_Score (%)': final_score,
    'Pass/Fail': pass_fail
})

# Save the dataset to a CSV file inside the "data" folder
df.to_csv("E:/PCCOE/Semesters/6th/ML/Mini Project/Dataset/student_data.csv", index=False)
