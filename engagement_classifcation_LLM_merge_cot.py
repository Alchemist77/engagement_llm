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
motion_df = pd.read_csv("data/combined_motion_data.csv")
emotion_df = pd.read_csv("data/combined_emotion_data.csv")
cognitive_df = pd.read_csv("data/combined_cognitive_data.csv")


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
model_path = "../../Qwen2.5-Coder-3B-Instruct"

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
NUM_TRIALS = 1  # Run the model 10 times for each interaction
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

    ### STEP-BY-STEP ANALYSIS:
        - **Step 1 (Motion Analysis)**: Evaluate movement smoothness, reaction time, and the fluidity of the human’s motion during the handover interaction.
        - If reaction time is low and movement is smooth, behavioral engagement is high.
        - If movements are abrupt or delayed, behavioral engagement is lower.

        - **Step 2 (Emotional Analysis)**: Evaluate facial valence and arousal levels.
        - If positive valence is high and arousal is moderate, affective engagement is high.
        - If negative valence is high, affective engagement is low.

        - **Step 3 (Cognitive Analysis)**: Consider gaze fixation and head movement.
        - If the user is consistently looking at the robot, cognitive engagement is high.
        - If gaze is unstable, cognitive engagement is lower.

        - **Step 4 (Final Engagement Classification)**: 
        - Assign engagement scores (1, 2, or 3) for each aspect.
        - Sum up the scores to get the **total engagement score**.
        - Assign the engagement level based on these scores:
            - **HIGH ENGAGEMENT**: 7-9
            - **MIDDLE ENGAGEMENT**: 5-6
            - **LOW ENGAGEMENT**: 3-4


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
            max_new_tokens=2000,
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

    # Directly take the single LLM response (since NUM_TRIALS = 1)
    llm_output = iteration_results[0]  # Directly access the first (and only) response

    # Extract engagement scores
    cognitive_score = llm_output.get("cognitive_score", 1)  # Default to 1 if missing
    affective_score = llm_output.get("affective_score", 1)
    behavioral_score = llm_output.get("behavioral_score", 1)

    # Compute total engagement score
    total_engagement = cognitive_score + affective_score + behavioral_score

    # Determine engagement level
    if total_engagement >= 7:
        engagement_level = "High"
    elif 5 <= total_engagement <= 6:
        engagement_level = "Middle"
    else:
        engagement_level = "Low"

    # Extract reasoning
    reasoning = llm_output.get("reasoning", "No reasoning provided.")

    # Store results
    engagement_results.append({
        'person': person_id,
        'interaction_order': interaction_order,
        'cognitive_score': cognitive_score,
        'affective_score': affective_score,
        'behavioral_score': behavioral_score,
        'total_engagement': total_engagement,
        'engagement_level': engagement_level,
        'reasoning': reasoning
    })

    print(f"Final result for Person {person_id}, Interaction {interaction_order}: {engagement_level}")


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
print("  - Engagement summary: results/cot_merge_engagement_summary.csv")
print("  - Engagement details: results/cot_merge_engagement_details.csv (includes reasoning)")
