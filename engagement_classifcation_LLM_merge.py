import torch
import pandas as pd
import json
import re
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter, defaultdict

# --------------------------
# Step 1: Load and Merge Data
# --------------------------

# Load CSV files for motion and emotion data
motion_df = pd.read_csv("data/test_combined_motion_data.csv")
emotion_df = pd.read_csv("data/test_combined_emotion_data.csv")
cognitive_df = pd.read_csv("data/test_combined_cognitive_data.csv")


# First, merge motion and emotion data on 'person' and 'v_order'
merged_df = pd.merge(motion_df, emotion_df, on=["person", "v_order"], how="inner")

# Then, merge the result with cognitive data
merged_df = pd.merge(merged_df, cognitive_df, on=["person", "v_order"], how="inner")

# Check the first few rows to confirm merging worked
#merged_df.to_csv("results/merge.csv", index=False)

# --------------------------
# Step 2: Load the LLM Model
# --------------------------

# Path to the downloaded model directory
model_path = "../Qwen2.5-Coder-3B-Instruct"

# Load the tokenizer and model with GPU allocation
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda"
)

# --------------------------
# Step 3: Define Variable Descriptions
# --------------------------

variable_descriptions = {
    # General Information
    "person": "Participant ID.",
    "v_order": "Interaction trial number.",

    # Interaction Timing
    "duration_interaction": "Time elapsed from video recording start to the end of release movement.",
    "duration_motion": "Time from the first movement in the reaching phase to object release.",
    "delta_starting_time": "[IGNORED] Time difference adjustment.",

    # Reaching Phase (Motion)
    "movement_time_REACH": "Time elapsed from the first movement of the reaching phase until the object is grasped.",
    "max_distance_REACH": "[IGNORED] Maximum reach distance.",
    "peak_vel_REACH": "Maximum velocity along the y-axis (axis of motion) in the reaching phase.",
    "time_to_peak_REACH": "Time elapsed from the first movement in reaching phase until peak velocity.",
    "reaction_REACH": "Time elapsed from the start of the interaction to the first reach movement.",

    # Release Phase (Motion)
    "movement_time_RELEASE": "Time elapsed from grasping to releasing the object.",
    "max_distance_RELEASE": "[IGNORED] Maximum release distance.",
    "peak_vel_RELEASE": "Maximum velocity along the x-axis (axis of motion) in the release phase.",
    "time_to_peak_RELEASE": "Time elapsed from the first movement in the release phase until peak velocity.",

    # Acceleration & Smoothness (Reaching)
    "avg_acc_REACH": "Average acceleration in the reaching phase.",
    "max_acc_REACH": "Maximum acceleration in the reaching phase.",
    "rmse_jerk_REACH": "Variability of the rate of change of acceleration in the reaching phase (smoothness).",
    "normalized_mean_velocity_REACH": "Ratio of max velocity to mean velocity in reaching phase (smoothness).",
    "number_movements_onset_REACH": "Number of frames where motion exceeds 5% of max velocity in reach interval.",
    "numberVelocityPeaks_REACH": "Number of velocity peaks above 10% of peak velocity in the reach movement.",

    # Acceleration & Smoothness (Release)
    "avg_acc_RELEASE": "Average acceleration in the release phase.",
    "max_acc_RELEASE": "Maximum acceleration in the release phase.",
    "rmse_jerk_RELEASE": "Variability of the rate of change of acceleration in the release phase (smoothness).",
    "normalized_mean_velocity_RELEASE": "Ratio of max velocity to mean velocity in the release phase (smoothness).",
    "number_movements_onset_RELEASE": "Number of frames where motion exceeds 5% of max velocity in the release interval.",
    "numberVelocityPeaks_RELEASE": "Number of velocity peaks above 10% of peak velocity in the release movement.",

    # Reaction Time
    "reactionTime_fromAnnotation": "[IGNORED] External annotation reaction time.",

    # Cognitive Aspects (Eye Gaze & Head Rotation)
    "total_blink_PM": "[IGNORED] Total number of blinks before movement.",
    "total_blink_REACH": "[IGNORED] Total number of blinks during reaching.",
    "total_blink_RELEASE": "[IGNORED] Total number of blinks during release.",
    "rate_blink_PM": "[IGNORED] Blink rate before movement.",
    "rate_blink_REACH": "[IGNORED] Blink rate during reaching.",
    "rate_blink_RELEASE": "[IGNORED] Blink rate during release.",

    "mean_gazeX_PM": "Average eye gaze direction (left-right) before movement.",
    "mean_gazeX_REACH": "Average eye gaze direction (left-right) during reaching.",
    "mean_gazeX_RELEASE": "Average eye gaze direction (left-right) during release.",
    "std_gazeX_PM": "Standard deviation of eye gaze direction (left-right) before movement.",
    "std_gazeX_REACH": "Standard deviation of eye gaze direction (left-right) during reaching.",
    "std_gazeX_RELEASE": "Standard deviation of eye gaze direction (left-right) during release.",

    "mean_gazeY_PM": "Average eye gaze direction (up-down) before movement.",
    "mean_gazeY_REACH": "Average eye gaze direction (up-down) during reaching.",
    "mean_gazeY_RELEASE": "Average eye gaze direction (up-down) during release.",
    "std_gazeY_PM": "Standard deviation of eye gaze direction (up-down) before movement.",
    "std_gazeY_REACH": "Standard deviation of eye gaze direction (up-down) during reaching.",
    "std_gazeY_RELEASE": "Standard deviation of eye gaze direction (up-down) during release.",

    "mean_headX_PM": "Average head rotation (pitch) before movement.",
    "mean_headX_REACH": "Average head rotation (pitch) during reaching.",
    "mean_headX_RELEASE": "Average head rotation (pitch) during release.",
    "std_headX_PM": "Standard deviation of head rotation (pitch) before movement.",
    "std_headX_REACH": "Standard deviation of head rotation (pitch) during reaching.",
    "std_headX_RELEASE": "Standard deviation of head rotation (pitch) during release.",

    "mean_headY_PM": "Average head rotation (yaw) before movement.",
    "mean_headY_REACH": "Average head rotation (yaw) during reaching.",
    "mean_headY_RELEASE": "Average head rotation (yaw) during release.",
    "std_headY_PM": "Standard deviation of head rotation (yaw) before movement.",
    "std_headY_REACH": "Standard deviation of head rotation (yaw) during reaching.",
    "std_headY_RELEASE": "Standard deviation of head rotation (yaw) during release.",

    # Emotional Valence (Positive & Negative)
    "mean_ps_PM": "Average positive valence before movement.",
    "mean_ps_REACH": "Average positive valence during the reaching phase.",
    "mean_ps_RELEASE": "Average positive valence during the release phase.",
    "std_ps_PM": "Standard deviation of positive valence before movement.",
    "std_ps_REACH": "Standard deviation of positive valence during reaching.",
    "std_ps_RELEASE": "Standard deviation of positive valence during release.",

    # Emotional Arousal (Positive & Negative)
    "mean_pa_PM": "Average positive arousal before movement.",
    "mean_pa_REACH": "Average positive arousal during the reaching phase.",
    "mean_pa_RELEASE": "Average positive arousal during the release phase.",
    "std_pa_PM": "Standard deviation of positive arousal before movement.",
    "std_pa_REACH": "Standard deviation of positive arousal during reaching.",
    "std_pa_RELEASE": "Standard deviation of positive arousal during release.",

    # Emotional Valence (Negative)
    "mean_ns_PM": "Average negative valence before movement.",
    "mean_ns_REACH": "Average negative valence during the reaching phase.",
    "mean_ns_RELEASE": "Average negative valence during the release phase.",
    "std_ns_PM": "Standard deviation of negative valence before movement.",
    "std_ns_REACH": "Standard deviation of negative valence during reaching.",
    "std_ns_RELEASE": "Standard deviation of negative valence during release.",

    # Emotional Arousal (Negative)
    "mean_na_PM": "Average negative arousal before movement.",
    "mean_na_REACH": "Average negative arousal during the reaching phase.",
    "mean_na_RELEASE": "Average negative arousal during the release phase.",
    "std_na_PM": "Standard deviation of negative arousal before movement.",
    "std_na_REACH": "Standard deviation of negative arousal during reaching.",
    "std_na_RELEASE": "Standard deviation of negative arousal during release.",

    # Source File References
    "source_file_x": "File reference for motion data.",
    "source_file_y": "File reference for emotion data."
}


# --------------------------
# Step 4: Process Interactions by Person
# --------------------------

# Create directories for outputs
os.makedirs("results", exist_ok=True)
os.makedirs("results/by_person", exist_ok=True)

# Store results organized by person
person_results = defaultdict(list)
NUM_TRIALS = 5  # Run the model 10 times for each interaction
engagement_results = []

# Loop through each row (interaction)
for idx, row in merged_df.iterrows():
    person_id = row["person"]
    interaction_order = row["v_order"]
    
    print(f"Processing Person {person_id}, Interaction {interaction_order}...")
    
    # Format interaction data
    interaction_data = row.to_dict()
    formatted_values = [
        f"{key}: {value} ({variable_descriptions.get(key, 'No description')})"
        for key, value in interaction_data.items()]
    interaction_text = "\n".join(formatted_values)
    
    # Create the LLM prompt
    prompt_text = f"""
    THIS DATA REPRESENTS HUMAN-ROBOT INTERACTION DURING AN OBJECT HANDOVER TASK. 
    Each row corresponds to a single interaction where a human hands over an object to a robot. 
    The dataset contains **motion characteristics** (e.g., speed, acceleration) and **emotional responses** (e.g., valence and arousal).
    THE GOAL IS TO ASSESS ENGAGEMENT BASED ON THREE ASPECTS:

    1. **Cognitive aspect**: "How much is the user watching the robot?" (Score: 3, 2, or 1)
    2. **Affective aspect**: "How happy is the user in the interaction?" (Score: 3, 2, or 1)
    3. **Behavioral aspect**: "How synchronized is the user’s movement with the robot?" (Score: 3, 2, or 1)

    ### Person {person_id}, Interaction {interaction_order} Data:
    {interaction_text}

    ### TASK:
    - Assign a score (1, 2, or 3) for each aspect.
    - Labels: 3 (Very much), 2 (not so much), 1 (none)​
    - Sum up the three scores to get the **total engagement score**.
    - Categorize the engagement level **STRICTLY based on this range**:
    - **HIGH ENGAGEMENT**: total engagement score **MUST be between 7 and 9**.
    - **MIDDLE ENGAGEMENT**: total engagement score **MUST be between 5 and 6**.
    - **LOW ENGAGEMENT**: total engagement score **MUST be between 3 and 4**.
    - YOU MUST ENSURE that the `"engagement_level"` directly matches the calculated `total_engagement` using the provided range.
    - Provide a brief explanation (reasoning) for why this classification was assigned.


    ### OUTPUT FORMAT:
    YOU MUST RESPOND ONLY WITH THE FOLLOWING JSON FORMAT AND NOTHING ELSE:
    
    {{
      "cognitive_score": 3 or 2 or 1,
      "affective_score": 3 or 2 or 1,
      "behavioral_score": 3 or 2 or 1,
      "total_engagement": cognitive_score + affective_score + behavioral_score,
      "engagement_level": "High(from 7 to 9)/Middle (from 5 to 6)/Low(from 3 to 4)",
      "reasoning": "Brief explanation based on motion and emotion data."
    }}
    """
    #print("prompt_text",prompt_text)

    # Store all results for this interaction
    iteration_results = []
    
    for iteration in range(NUM_TRIALS):
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        try:
            json_matches = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', response, re.DOTALL)
            if json_matches:
                parsed_json = json.loads(json_matches[-1])
                iteration_results.append(parsed_json)
            else:
                iteration_results.append({
                    "cognitive_score": 0,
                    "affective_score": 0,
                    "behavioral_score": 0,
                    "total_engagement": 0,
                    "engagement_level": "ERROR",
                    "reasoning": "No JSON response."
                })
        except json.JSONDecodeError:
            iteration_results.append({
                "cognitive_score": 0,
                "affective_score": 0,
                "behavioral_score": 0,
                "total_engagement": 0,
                "engagement_level": "ERROR",
                "reasoning": "Invalid JSON format."
            })

    # Determine the most common total engagement and level
    #print("iteration_results",iteration_results)
    total_scores = [res["total_engagement"] for res in iteration_results]
    #print("total_scores",total_scores)

    most_common_total, _ = Counter(total_scores).most_common(1)[0]
    # Ensure the engagement level matches the calculated total engagement
    if most_common_total >= 7:
        most_common_level = "High"
    elif 5 <= most_common_total <= 6:
        most_common_level = "Middle"
    else:
        most_common_level = "Low"
    #print("most_common_level",most_common_level)


    # Ensure selection of most common individual scores
    cognitive_scores = next((res["cognitive_score"] for res in iteration_results if res["total_engagement"] == most_common_total), "N/A")
    affective_scores = next((res["affective_score"] for res in iteration_results if res["total_engagement"] == most_common_total), "N/A")
    behavioral_scores = next((res["behavioral_score"] for res in iteration_results if res["total_engagement"] == most_common_total), "N/A")


    # Get reasoning from the most common total engagement score
    matching_entries = [res for res in iteration_results if res["total_engagement"] == most_common_total]
    selected_reasoning = matching_entries[0]["reasoning"] if matching_entries else "No reasoning provided."


    # Store results
    engagement_results.append({
        'person': person_id,
        'interaction_order': interaction_order,
        'cognitive_score': cognitive_scores,
        'affective_score': affective_scores,
        'behavioral_score': behavioral_scores,
        'total_engagement': most_common_total,
        'engagement_level': most_common_level,
        'reasoning': selected_reasoning
    })

    print(f"Final result for Person {person_id}, Interaction {interaction_order}: {most_common_level}")

# --------------------------
# Step 5: Save Results
# --------------------------

# Convert results to DataFrame
engagement_df = pd.DataFrame(engagement_results)

# Save basic engagement levels (without reasoning)
engagement_df[['person', 'interaction_order', 'total_engagement', 'engagement_level']].to_csv(
    "results/merge_engagement_summary.csv", index=False
)

# Save detailed engagement levels with reasoning
engagement_df.to_csv("results/merge_engagement_details.csv", index=False)

print("Analysis complete!")
print("  - Engagement summary: results/merge_engagement_summary.csv")
print("  - Engagement details: results/merge_engagement_details.csv (includes reasoning)")