## Monopoly Gameplay Recognition System

## Check out the full report: [link](https://drive.google.com/file/d/1zC359NF-jmL3sk3g6ifQ08YzqiRZDEBX/view?usp=sharing)

This project implements a computer vision system designed to recognize and track the state of gameplay in the board game Monopoly (Classic Edition). The system analyzes overhead video recordings to track game elements, detect player interactions, and log specific game events such as pawn movements and dice rolls.

**Key Constraint**: The solution relies entirely on classical computer vision techniques (OpenCV) without the use of neural networks.

### 1. Methodology & Pipeline
The system utilizes a modular architecture where specific scripts handle the detection of different game elements.

**A. Board Localization (Homography)**
To standardize detection regardless of camera angle, the system localizes the board using edge detection and contour analysis.
Technique: Gaussian blurring and Canny edge detection (thresholds 60/160) identify prominent edges.
Warping: A Homography matrix is calculated to warp the board into a fixed 900×900 pixel top-down view, allowing for the definition of static Regions of Interest (ROI) for fields and the center box.

**B. Hand Detection (Game Flow Control)**
To prevent false detections during player moves, the system detects the presence of hands to pause state updates.
Fusion Approach: Combines Skin Color Segmentation (HSV color space) with Motion Detection (temporal difference against a running average background).
Logic: If the intersection of skin pixels and motion mask exceeds a threshold, the event HAND_APPEARS is triggered.

**C. Pawn Detection and Tracking**
Pawn tracking is achieved through background subtraction, comparing the current frame against a "clean" reference image.
Background Extraction: The system uses a static "clean" board image and subtracts it from the current frame to isolate foreground objects.
Filtering: Resulting blobs are filtered by Solidity, Extent, and Aspect Ratio to ensure only pawn-shaped objects are tracked.
Mapping: Centroids of detected blobs are mapped to one of the 40 distinct board fields.

**D. Dice and Card Detection**
Dice: Detected in the "Inner Box" using LAB color space segmentation (isolating high lightness/neutral color). Pips are counted using adaptive thresholding and Connected Component Analysis.
Cards: Identified using geometric filtering (Gaussian blur + Canny) to find rectangular contours in the center field.

### 2. Game State Logic
An orchestrator pipeline integrates individual detections into a cohesive "Game Status" state machine.
**State Vector**: Per frame, the system tracks Hand presence (T/F), Dice values, and Pawn field IDs.

**Event Triggers**:
- HAND_APPEARS / HAND_DISAPPEARS: Gates the logic to prevent noise during interaction.
- DICE_STABLE: Triggered when dice values remain identical for 8 consecutive frames.
- PAWN_MOVED: Triggered only when a pawn's field changes, the hand is not present, and the new position is stable.

### 3. Performance Analysis
The system was tested on datasets categorized as Easy (stable light), Medium (shadows), and Difficult (camera shake/angles).

| Feature         | Accuracy / Success Rate     | Notes |
|----------------|-----------------------------:|-------|
| Hand Detection | 100% (31/31 events)          | The strongest feature; successfully filtered interactions across all difficulty levels. |
| Dice Stability | ~90%                         | Reliably detects when dice are rolled, though reading specific pip values was 0–40% effective due to contrast issues. |
| Card Detection | 50–100%                      | Generally accurate, occasional occlusion by shadows. |
| Pawn Tracking  | 17–60%                       | Highly sensitive to lighting. The "light grey" pawn frequently blended with the background, while the red pawn was detected consistently. |

### 4. Conclusion
The system successfully implements a logic engine that tracks the main flow of Monopoly. While the Hand Detection and Board Warping proved highly robust, the Background Subtraction method for pawns demonstrated the limitations of classical CV in uncontrolled lighting environments, particularly regarding contrast sensitivity.

**Technologies:** Python, OpenCV, NumPy.
