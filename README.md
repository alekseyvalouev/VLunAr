# VLunAr: Vision-Language-Action Lunar Lander

Vision-Language-Action (VLA) model capable of piloting a lunar lander using natural-language instructions. This project integrates reinforcement learning, deep vision models, and language-conditioned action generation to build an intelligent lunar-mission control system.

---

## Project Overview

VLunAr aims to create a model that can interpret natural language commands—ranging from precise instructions  
(e.g., “Ascend 1 meter, then land 1 meter to the right”)  
to broad goals  
(e.g., “Fly to the red flag”)  
and execute them from any initial lander configuration.

Because modern VLA models require large labeled datasets, this project first trains a suite of Deep Q-Networks (DQNs) to master individual subtasks. These DQNs then serve as automated experts to label examples that power the VLA training stage.

---

## Mentor
- Sandeep

---

## Literature Review

### Reinforcement Learning
- Deep Q-Networks (DQNs)  
- Q-learning  
- Gymnasium Reinforcement Learning Environments  

### Vision-Language-Action Models
- OpenVLA: An Open-Source Vision-Language-Action Model  
- PISTAR06 post-training technique:  
  https://www.physicalintelligence.company/blog/pistar06  
- Robotics foundation models overview:  
  https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models

---

## Implementation Plan

### 1. Environment and Data
We will use the Gymnasium Lunar Lander (Box2D) environment:  
https://gymnasium.farama.org/environments/box2d/lunar_lander/#rewards

A custom 224×224 RGB renderer will be implemented for VLA visual inputs.

---

### 2. DQN Training for Subtasks
We will train multiple Deep Q-Networks to perform key behaviors:

- Hover  
- Strafe  
- Takeoff  
- Land  
- Navigate to a target or flag  

These trained policies will serve as automated expert demonstrators.

---

### 3. Dataset Construction
After each DQN achieves stable performance:

- Collect [X AMOUNT] expert trajectories per task  
- Ensure diverse starting states  
- Pair each sequence with appropriate language instructions  
- Produce a large dataset suitable for VLA training

---

### 4. Training the VLA Model

#### Architecture
- Vision encoder: DINOv2 (frozen)  
- Language encoder: frozen  
- Action head: trainable  

#### Steps
1. Modify OpenVLA to support the Lunar Lander action space.  
2. Ensure efficient image data throughput for training.  
3. Train the VLA using DQN-labeled examples, learning the mapping  
   `(image, language instruction) → optimal action`.

---

### 5. Post-Training Improvements
- Apply PISTAR06-style post-training  
- Allow expert demonstration corrections  
- Fine-tune based on failure cases  

---

### 6. Evaluation
Evaluate the VLA on:

- Hovering  
- Strafing  
- Takeoff  
- Landing  
- Navigation tasks  

Also evaluate generalization to novel lander states and new natural-language command variations.

---

## Action Items (Project TODO)

- [ ] Port 224×224 renderer  
- [ ] Train DQN for hover task  
- [ ] Train DQN for strafe task  
- [ ] Train DQN for takeoff task  
- [ ] Train DQN for land task  
- [ ] Train DQN for navigation-to-flag task  
- [ ] Modify OpenVLA to support the Lunar Lander action space  
- [ ] Ensure image streaming pipeline is efficient  
- [ ] Train the VLA on DQN-generated data  

---

## License
MIT License (or institution-specific license to be determined)

---

## Acknowledgements
Thanks to mentor Sandeep, and to the developers of OpenVLA, Gymnasium, and DINOv2 for enabling this project.

