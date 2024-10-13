# Week 2
## Block-based and Multiresolutions histograms
<img src="https://github.com/MCV-2024-C1-Project/Team5/blob/main/week2/assets/block_based_histograms/block_based_diagram.png"/>

### Evaluation

| Comparison metric  | Histogram  | Level        | mapk1    | mapk5    |
|--------------------|------------|--------------|----------|----------|
| L1                 | 1D         | 12           | 0.90     | 0.916    |
| L1                 | 1D         | [12, 20]     | 0.90     | 0.916    |
| Bhattacharyya      | 1D         | 20           | 0.90     | 0.908    |
| Bhattacharyya      | 1D         | [18, 19, 20, 21]     | 0.83     | 0.891    |
| Bhattacharyya      | 2D         | [18, 19, 20, 21]     | 0.83     | 0.83    |
| L1      | 2D         | 14           | 0.86     | 0.87    |
| Bhattacharyya      | 3D         | 20           | 0.70     | 0.71    |
| L1      | 3D         | 12         | 0.70     | 0.71    |


We were able to observe how 1D histograms perform way better. Also, if we stack different levels in the spatial pyramid, they don't increase performance.

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

6. Apply an **opening by reconstruction of erosion** to achieve a hole filling and remove interior black holes in the paintings.
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
| **Precision** | 0.91     |
| **Recall**    | 0.94     |
| **F1-Score**  | 0.91     |
