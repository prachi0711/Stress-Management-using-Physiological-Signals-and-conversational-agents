# Dialogue Manager 

The Dialogue Manager (`dialogue_manager.py`) connects stress classification with user-facing interaction. It uses uncertainty scores to decide when to offer support and when to ask for feedback.

### Features

* Tracks user states across interactions.
* Uses uncertainty threshold (default = 0.45) to guide dialogue.
* Adapts responses based on classification (stress/calm), uncertainty, and feedback.
* Supports feedback loops (e.g., user says “stressed” → offers breathing exercise).

---

### Dialogue Logic

| Condition                                     | System Response                                                                                   | Dialogue State                   |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------- |
| **Entropy < threshold & pred = calm**         | “You seem calm – nice. Keep doing what you're doing!”                                             | Normal                           |
| **Entropy < threshold & pred = stress**       | “You seem stressed. Would you like a short breathing exercise?”                                   | Offer help, waiting for feedback |
| **Entropy > threshold (uncertain, <3 times)** | “I’m getting mixed signals about your stress level. Let me observe a bit more.”                   | Uncertain state                  |
| **Entropy > threshold (≥3 times)**            | “I am uncertain about your stress level. Can you please tell me how you feel?”                    | Asks for user feedback           |
| **User feedback = yes (help)**                | “Great! Let's begin with a 30-second breathing exercise. Breathe in… and out…”                    | Starts guided activity           |
| **User feedback = no (help)**                 | “Okay, no problem. Let me know if you change your mind.”                                          | Ends help offer                  |
| **User feedback = stressed**                  | “Thanks for telling me. Let’s try a short breathing exercise together. Would you like to try it?” | Offer help again                 |
| **User feedback = calm**                      | “Good to hear that you feel calm. Keep it up!”                                                    | Positive reinforcement           |
| **User feedback = need\_help**                | “Alright, I’ll suggest a quick stress-relief activity. Would you like a breathing exercise?”      | Offer help                       |
| **User feedback = no\_help**                  | “Okay, I’ll give you some space. Let me know if you need support later.”                          | End interaction                  |
| **Other feedback**                            | “Thanks for sharing. I’ll keep that in mind.” / "I didn't understand your response. Please say 'yes' or 'no' if you'd like a breathing exercise." | Neutral acknowledgment  / Asking for Clarification|

Next Step: [ROS2 Node](https://github.com/prachi0711/Stress-Management-using-Physiological-Signals-and-conversational-agents/blob/main/ros_node/README.md)

