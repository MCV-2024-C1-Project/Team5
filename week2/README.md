# Week 2
## Task 1 - Block and hierarchical histograms
...

## Background removal
In order to extract the mask and foreground, we follow the next steps:
### Phase 1: Image Preprocessing
1. Convert the image to **grayscale** to simplify the image data for easier processing and apply **Gaussian blur** to reduce noise and smooth the image.
<p align="center">
  <img src="https://github.com/MCV-2024-C1-Project/Team5/blob/main/week2/assets/background_removal/step_1.png" width="600"/>
</p>

2. Use **Otsu’s thresholding** to determine the optimal threshold to create a binary mask.
<p align="center">
  <img src="https://github.com/MCV-2024-C1-Project/Team5/blob/main/week2/assets/background_removal/step_2.png" width="600"/>
</p>

3. **Invert** the binary mask and identify the foreground.
<p align="center">
  <img src="https://github.com/MCV-2024-C1-Project/Team5/blob/main/week2/assets/background_removal/step_3.png" width="600"/>
</p>

### Phase 2: Mask Refinement
4. Apply **morphological closing** to fill the small black holes and remove noise using a 15x15 kernel. We want to unify small unconnected components that should belong together.
<p align="center">
  <img src="https://github.com/MCV-2024-C1-Project/Team5/blob/main/week2/assets/background_removal/step_4.png" width="600"/>
</p>

5. Identify the **largest connected component** and retain the largest connected component to create a new mask.
<p align="center">
  <img src="https://github.com/MCV-2024-C1-Project/Team5/blob/main/week2/assets/background_removal/step_5.png" width="600"/>
</p>

6. Apply a 40x40 **morphological closing** to fill black holes in the image, as no risk of connecting unconnected components, as far as kernel doesn’t touch image boundaries.
<p align="center">
  <img src="https://github.com/MCV-2024-C1-Project/Team5/blob/main/week2/assets/background_removal/step_6.png" width="600"/>
</p>

### Phase 3: Evaluation
To evaluate the performance of the subtracted masks, we use three key metrics: **Precision**, **Recall**, and **F1-Score**.

#### 1. Precision
Corresponds to the ratio of correctly predicted positive cases to the total predicted positive cases.
- **Formula**: `Precision = TP / (TP + FP)`
> **Note:**  
> Of all the instances classified as positive, how many are actually positive?

#### 2. Recall
Measures the proportion of actual positives that were correctly identified.
- **Formula**: `Recall = TP / (TP + FN)`
> **Note:**  
> Of all the positive instances, how many are correctly identified?
#### 3. F1-Score
Mean of Precision and Recall.
- **Formula**: `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

### Results
| Metric        | Value    |
|---------------|----------|
| **Precision** | 0.87677  |
| **Recall**    | 0.7769   |
| **F1-Score**  | 0.8077   |

## Task 5 - Results
...
