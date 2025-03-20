# ML 24/25-01 Investigate Image Reconstruction By using Classifier
###### Through out this project we contribute to  implement the image reconstruction using classifiers in NeoCortexAPI

[![N|Logo](https://ddobric.github.io/neocortexapi/images/logo-NeoCortexAPI.svg )](https://ddobric.github.io/neocortexapi/)

## Introduction

Image reconstruction is a critical task in computer vision, where missing or degraded images are restored using machine learning techniques. This study explores the use of classifiers—K-Nearest Neighbors (KNN) and Hierarchical Temporal Memory (HTM)—for reconstructing images from Sparse Distributed Representations (SDRs). By leveraging NeoCortexApi, the system processes binarized images, classifies them, and reconstructs original structures while assessing accuracy using Cosine similarity metrics.

# Methodology

*Fig: Methodology Flowchart*
(Adding Block diagram)
The image reconstruction pipeline consists of multiple stages: image binarization, SDR encoding, classification, and reconstruction. 

## Overview

This project explores image reconstruction using **Hierarchical Temporal Memory (HTM)** and **K-Nearest Neighbors (KNN)** classifiers. The core idea is to binarize input images, process them into **Sparse Distributed Representations (SDRs)** using a Spatial Pooler, classify the SDRs, and reconstruct images from the predicted SDRs. The reconstructed images are then compared using similarity metrics.

## Key Components

### 1. **Image Binarization**

Images are converted into binary matrices (0s and 1s) before processing.

```csharp
public static int[,] BinarizeImage(Bitmap image, int threshold = 128) {
    int width = image.Width;
    int height = image.Height;
    int[,] binarizedImage = new int[width, height];
    
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            Color pixel = image.GetPixel(x, y);
            int gray = (pixel.R + pixel.G + pixel.B) / 3;
            binarizedImage[x, y] = gray > threshold ? 1 : 0;
        }
    }
    return binarizedImage;
}
```

### 2. **Generating SDRs using the Spatial Pooler**

The **Spatial Pooler** converts binarized images into **Sparse Distributed Representations (SDRs)**.

```csharp
SpatialPooler sp = new SpatialPooler();
sp.Compute(inputArray, activeArray);
```

### 3. **Classification using HTM and KNN**

#### HTM Classifier Training & Prediction

```csharp
HTMClassifier classifier = new HTMClassifier();
classifier.Learn(trainingSDR, label);
var predictedSDR = classifier.Predict(testSDR);
```

#### KNN Classifier Training & Prediction

```csharp
KNNClassifier knn = new KNNClassifier(k: 5);
knn.Train(trainingSDRs, trainingLabels);
var predictedSDR = knn.Predict(testSDR);
```

### 4. **Image Reconstruction**

The predicted SDRs are used to reconstruct images.

```csharp
public Bitmap ReconstructImage(int[] predictedSDR, int width, int height) {
    Bitmap reconstructed = new Bitmap(width, height);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int value = predictedSDR[x + y * width] > 0 ? 255 : 0;
            reconstructed.SetPixel(x, y, Color.FromArgb(value, value, value));
        }
    }
    return reconstructed;
}
```

### 5. **Similarity Calculation**

We compare original and reconstructed images using **Adjusted Jaccard Similarity** (HTM) and **Cosine Similarity** (KNN).

#### Adjusted Similarity (HTM)

```csharp
public static double AdjustedJaccardSimilarity(int[] a, int[] b) {
    int intersection = a.Zip(b, (x, y) => x & y).Sum();
    int union = a.Zip(b, (x, y) => x | y).Sum();
    return union == 0 ? 0 : (double)intersection / union;
}
```

#### Cosine Similarity (KNN)

```csharp
public static double CosineSimilarity(int[] a, int[] b) {
    double dot = a.Zip(b, (x, y) => x * y).Sum();
    double magA = Math.Sqrt(a.Sum(x => x * x));
    double magB = Math.Sqrt(b.Sum(y => y * y));
    return magA == 0 || magB == 0 ? 0 : dot / (magA * magB);
}
```

### 6. **Results Visualization**

The similarity metrics are plotted using CSV data.

```csharp
public void GenerateSimilarityGraph() {
    var data = LoadCSV("similarity_results.csv");
    PlotGraph(data);
}
```

## Running the Project

1. **Prepare Data:** Ensure images are binarized.
2. **Train Classifiers:** Use training images to learn SDR patterns.
3. **Predict SDRs:** Run the trained classifiers on test images.
4. **Reconstruct Images:** Convert predicted SDRs back into images.
5. **Evaluate Results:** Compare reconstructed images with originals.

This project demonstrates how **HTM** and **KNN** classifiers can be used for **image reconstruction** based on SDRs. The similarity metrics provide insights into the effectiveness of each method.
# Implementation
## **HTM and KNN Classifier Training Process**

This document describes the training process for the **HTM (Hierarchical Temporal Memory) Classifier** and **KNN (K-Nearest Neighbors) Classifier** using image-based **Sparse Distributed Representations (SDRs)**.
The training phase processes multiple images to generate SDRs and train both classifiers. The process continues until the system reaches a stable state or completes the maximum training cycles.

### **Training Loop**
The training loop runs until either:
- The system reaches a **stable state**, or
- The **maximum training cycles** (`maxCycles`) are completed.

### **Steps in Each Training Cycle**
For each image, the following steps are performed:

1. **Binarization**: Convert the image into a binary representation.
   ```csharp
   string binarizedImageFile = BinarizeImageToFixedSize(image, imgSize);
   ```
   - Resizes the image and converts it into a binary format.
   - Saves the binary image as a text file.

2. **Read Binary Representation**: Read the binary representation from the text file.
   ```csharp
   int[] inputVector = ReadBinaryTextFile(binarizedImageFile);
   ```

3. **Compute Active Columns Using Spatial Pooler**:
   ```csharp
   sp.compute(inputVector, activeArray, true);
   var activeCols = ArrayUtils.IndexWhere(activeArray, el => el == 1);
   ```
   - Converts the binary input into an SDR.
   - Extracts active columns representing key image features.

4. **Convert Active Columns to SDR Format**:
   ```csharp
   var activeCells = activeCols.Select(colIdx => new NeoCortexApi.Entities.Cell { Index = colIdx }).ToArray();
   ```

5. **Train the KNN Classifier**:
   ```csharp
   knnClassifier.Learn(image, activeCells);
   ```

6. **Train the HTM Classifier**:
   ```csharp
   Htmclassifier.Learn(image, activeCols);
   ```

7. **Store SDR in Dictionary for Later Retrieval**:
   ```csharp
   if (!storedSDRs.ContainsKey(image))
   {
       storedSDRs[image] = new List<int[]>();
   }
   storedSDRs[image].Add(activeCols);
   ```

8. **Save SDR to a File**:
   ```csharp
   string sdrFilePath = Path.Combine(sdrFolder, $"{Path.GetFileNameWithoutExtension(image)}.csv");
   File.WriteAllText(sdrFilePath, string.Join(",", activeCols));
   ```

9. **Handle Errors**:
   ```csharp
   catch (Exception ex)
   {
       Debug.WriteLine($" Error processing {image}: {ex.Message}");
   }
   ```

10. **Increment Training Cycle**:
    ```csharp
    currentCycle++;
    Debug.WriteLine($" Completed Cycle {currentCycle}.");
    ```

### **Checking Model Stability**
After training, the system checks whether a **stable state** has been reached:
```csharp
if (!isInStableState)
{
    Debug.WriteLine(" Training completed, but stable state not reached.");
}
else
{
    Debug.WriteLine(" Training completed successfully.");
}
```

---

## **Logging Predicted SDRs**
After training, the predicted SDRs are logged for both classifiers.

### **Logging KNN Predictions**
### Code Snippet
```csharp
Debug.WriteLine("\n--- PREDICTED KNN SDRs ---");
foreach (var sdr in predictedSDRsListKnn)
{
    Debug.WriteLine($"Predicted SDR: {string.Join(", ", sdr)}");
}
```

### **Logging HTM Predictions**
### Code Snippet
```csharp
Debug.WriteLine("\n--- PREDICTED HTM SDRs ---");
foreach (var sdr in predictedSDRsListHtm)
{
    Debug.WriteLine($"Predicted SDR: {string.Join(", ", sdr)}");
}
```

### **Logging Stored SDRs**
### Code Snippet
```csharp
Debug.WriteLine("\n--- STORED SDRs ---");
foreach (var label in storedSDRs.Keys)
{
    foreach (var storedSDR in storedSDRs[label])
    {
        Debug.WriteLine($"Label: {label}, SDR: {string.Join(", ", storedSDR)}");
    }
}

# Classification and Image Reconstruction

## Classification Phase  

This phase **classifies SDRs** using **K-Nearest Neighbors (KNN)** and **Hierarchical Temporal Memory (HTM)** classifiers.

### KNN-Based Classification
### Code Snippet

```csharp
 Debug.WriteLine("\n--- KNN CLASSIFICATION RESULTS ---");
            foreach (var sdr in predictedSDRsListKnn)
            {
                var activeCells = sdr.Select(index => new NeoCortexApi.Entities.Cell { Index = index }).ToArray();
                var predictionsKnn = knnClassifier.GetPredictedInputValues(activeCells, 4);

                Debug.WriteLine($"------------KNN Results-----------------");
                Debug.WriteLine($"\n SDR: {string.Join(", ", sdr)}");

                foreach (var prediction in predictionsKnn)
                {
                    Debug.WriteLine($" Predicted Label: {prediction.PredictedInput}");
                }
            }
```
- Converts each SDR into an array of active cells.  
- Uses **KNN classifier** to predict labels.  
- Logs the predicted labels for analysis.  

### HTM-Based Classification  
### Code Snippet
```csharp
Debug.WriteLine("\n--- HTM CLASSIFICATION RESULTS ---");
            foreach (var sdr in predictedSDRsListHtm)
            {
                var activeCells = sdr.Select(index => new NeoCortexApi.Entities.Cell { Index = index }).ToArray();
                var predictionsHtm = Htmclassifier.GetPredictedInputValues(activeCells, 4);

                Debug.WriteLine($"------------HTM Results-----------------");
                Debug.WriteLine($"\n SDR: {string.Join(", ", sdr)}");

                foreach (var prediction in predictionsHtm)
                {
                    Debug.WriteLine($" Predicted Label: {prediction.PredictedInput}");
                }
            }
```
- Converts each SDR into an array of active cells.  
- Uses **HTM classifier** to predict labels.  
- Logs the predicted labels for analysis.  

---

## Image Reconstruction Phase  

After classification, **image reconstruction** is performed using the **predicted SDRs**.
### Code Snippet
```csharp
RunRustructuringExperimentKNN(sp, predictedSDRsListKnn);
            Debug.WriteLine("\n Running KNN Restructuring Experiment...");

            /// <summary>
            /// Runs image reconstruction experiment using HTM classifier's SDRs.
            /// </summary>
            RunRustructuringExperimentHtm(sp, predictedSDRsListHtm);
            Debug.WriteLine("\n Running HTM Restructuring Experiment...");
```

### KNN-Based Image Reconstruction  
- Uses **KNN-predicted SDRs** to reconstruct images.  
- Evaluates how closely the reconstructed image matches the original input.  

### HTM-Based Image Reconstruction  
- Uses **HTM-predicted SDRs** to reconstruct images.  
- Analyzes how well the HTM classifier retained image patterns.  

## 4. Returning Trained Components & Results
This section describes the process of returning the trained classifiers, predicted SDRs, and stored SDRs for further processing. After training, both the **HTM Classifier** and **KNN Classifier** generate predicted SDRs, which are then used to reconstruct images and evaluate similarity.

### Returned Components
The following components are returned after training:
- **sp**: Trained Spatial Pooler instance.
- **knnClassifier**: KNN classifier instance with trained SDRs.
- **Htmclassifier**: HTM classifier instance with learned SDR patterns.
- **predictedSDRsListKnn**: List of predicted SDRs from the KNN classifier.
- **predictedSDRsListHtm**: List of predicted SDRs from the HTM classifier.
- **storedSDRs**: Collection of original SDRs used for classification.

### Code Snippet
```python
return (sp, knnClassifier, Htmclassifier, predictedSDRsListKnn, predictedSDRsListHtm, storedSDRs)
```
##  Results & Discussion

### **A. Image Reconstruction Performance**
The performance of HTM and KNN classifiers was evaluated by comparing the reconstructed images with the original binarized inputs. The reconstructed images were obtained by feeding predicted **Sparse Distributed Representations (SDRs)** into the inverse transformation process.

- **HTM Classifier** retains more structural details.
- **KNN Classifier** is faster but introduces slight distortions.



 output image




- **Results:**  
  - **The HTM classifier outperforms KNN in classification**, indicating that it is better at learning temporal patterns and making precise SDR predictions..
  - **Although KNN has slightly lower accuracy**, it is faster and simpler to implement.



 similarity graph




### **B. Classification Accuracy**
- **HTM classifier achieved higher classification accuracy**:
  - **HTM Accuracy**: 83.51%
  - **KNN Accuracy**: 80.12%
- **KNN classifier is computationally faster**, making it preferable for real-time applications.

### **C. Discussion**
1. **KNN classifier achieves higher similarity scores** and preserves pixel-level details.
2. **HTM classifier outperforms KNN in classification accuracy**, making it better for SDR learning.
3. **KNN classifier offers faster processing**, suitable for real-time applications.
4. **Future work** can explore **hybrid models** to combine HTM’s accuracy with KNN’s efficiency.

This study highlights the trade-offs between **HTM and KNN classifiers** for SDR-based image reconstruction, helping guide future improvements in biologically inspired machine learning.

# **Test Cases**
### CosineSimilarity_IdenticalVectors_ReturnsOne
- *Test Category:* CosineSimilarityIdenticalVectors
- *Description: * Verifies that the CosineSimilarity function returns 1.0 when comparing two identical vectors.

### CosineSimilarity_CompletelyDifferentVectors_ReturnsZero
- *Test Category:* CosineSimilarityDifferentVectors
- *Description: * Confirms that CosineSimilarity correctly returns 0.0 when given two completely dissimilar vectors.

### CosineSimilarity_PartiallySimilarVectors_ReturnsExpectedValue
- *Test Category:* CosineSimilarityPartialMatch
- *Description: * Ensures that the function returns a similarity score between 0.0 and 1.0 when given two partially similar vectors.


### ConvertPermanenceToBinary_ValidInput_ReturnsExpectedBinaryArray
- *Test Category:* PermanenceConversionValidInput
- *Description: * Ensures that the ConvertPermanenceToBinary function correctly converts permanence values into binary format using a 0.7 threshold.

### ConvertPermanenceToBinary_AllPositivePermanences_ReturnsExpectedValues
- *Test Category:* PermanenceConversionAllPositive
- *Description: * Verifies that the function correctly processes an array where all permanence values are non-negative.

### ConvertPermanenceToBinary_NegativeValue_ThrowsException
- *Test Category:* PermanenceConversionNegativeValues
- *Description: * Confirms that an ArgumentException is thrown when a permanence array contains negative values.


### AddActiveColsToStoredSDRs_StoresActiveColumnsCorrectly
- *Test Category:* SDRStorageValidation
- *Description: * Verifies that the AddActiveColsToStoredSDRs function correctly stores active column SDRs in a dictionary, ensuring accurate image reconstruction and classification.
