# Stress Management using Physiological Signals and Conversational Agents

Stress is a common issue that significantly impacts mental and physical health, often diminishing overall well-being and productivity. Effective stress management
is crucial, yet many individuals struggle to recognize or address their stress levels in a timely manner. Advances in technology, particularly artificial intelligence
(AI), offer promising solutions by leveraging physiological signals to detect and mitigate stress in real time. This project explores an AI-based approach to stress
management by analyzing physiological indicators such as heart rate and skin conductance, enabling early intervention through personalized, interactive support.

Physiological signals: measurements of biological functions such as heart rate, skin conductance, and brain activity; provide direct insight into how the body responds to stressors. A key innovation of this system is its integration of physiological monitoring with conversational interaction facilitated by a companion robot. Unlike traditional methods that rely on self-reported assessments, physiological signals provide continuous, objective, and involuntary measurements of stress responses, making them more reliable and difficult to manipulate. These signals reflect the body’s dynamic reactions to stressors, allowing for tailored stress-relief strategies based on individual needs.

This project leverages physiological signals (e.g., IBI from BVP, EDA) from the **WESAD** dataset to detect stress levels in real time. An AI pipeline classifies stress, quantifies prediction uncertainty, and triggers personalized interventions via a dialogue system.

# **Table of Contents**  
1. [Data Collection](https://github.com/prachi0711/Stress-Management-using-Physiological-Signals-and-conversational-agents/blob/main/Dataset/README.md) 
2. [Preprocessing](https://github.com/prachi0711/Stress-Management-using-Physiological-Signals-and-conversational-agents/blob/main/preprocessing/README.md)  
3. [Stress Classification](https://github.com/prachi0711/Stress-Management-using-Physiological-Signals-and-conversational-agents/blob/main/stress_classification/README.md) 
4. [Uncertainty Quantification](https://github.com/prachi0711/Stress-Management-using-Physiological-Signals-and-conversational-agents/blob/main/uncertainty_quantification/README.md) 
5. [Dialogue Manager](https://github.com/prachi0711/Stress-Management-using-Physiological-Signals-and-conversational-agents/blob/main/dialogue_manager/README.md)
6. [Ros Node](https://github.com/prachi0711/Stress-Management-using-Physiological-Signals-and-conversational-agents/blob/main/ros_node/README.md) 


# **Workflow Summary**  

  Data Collection: Use the WESAD dataset (BVP and EDA signals) for stress detection.

  Preprocessing: Filter noise, normalize, and extract features (e.g., IBI peaks, EDA).

  Stress Classification: Train a model (Random Forest/1D CNN) to predict stress.

  Uncertainty Quantification (UQ): Compute confidence scores (entropy) to assess prediction reliability.

  Decision Logic:

  - Low uncertainty: Proceed with predicted stress level.

  - High uncertainty: Prompt user for feedback (e.g., "How are you feeling?").

  Dialogue Management: Generate context-aware responses (e.g., breathing exercises for stress).

# **Environment Setup**

### Step 1: Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies:

```
python -m venv .venv
```
Activate the virtual environment:

Linux/macOS:
   ```
   source .venv/bin/activate
```

Windows:

    .\.venv\Scripts\activate
    
### Step 2: Install Dependencies

Install the required dependencies:
```
pip install -r requirements.txt
```

This will install necessary libraries like neurokit2, requests, and others.

To get more information about this project, please follow the [report]()
