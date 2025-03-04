using NeoCortex;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;
using NeoCortexApi;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NeoCortexApi.Classifiers;
using System.Text;

namespace NeoCortexApiSample
{
    internal class ImageBinarizerSpatialPattern
    {
        public string inputPrefix { get; private set; }

        /// <summary>
        /// Implements an experiment that demonstrates how to learn spatial patterns.
        /// SP will learn every presented Image input in multiple iterations.
        /// </summary>
        public void Run()
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(ImageBinarizerSpatialPattern)}");

            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            // We will build a slice of the cortex with the given number of mini-columns
            int numColumns = 64 * 64;
            // The Size of the Image Height and width is 28 pixel
            int imageSize = 28;
            var colDims = new int[] { 64, 64 };

            // This is a set of configuration parameters used in the experiment.
            HtmConfig cfg = new HtmConfig(new int[] { imageSize, imageSize }, new int[] { numColumns })
            {
                CellsPerColumn = 10,
                InputDimensions = new int[] { imageSize, imageSize },
                NumInputs = imageSize * imageSize,
                ColumnDimensions = colDims,
                MaxBoost = maxBoost,
                DutyCyclePeriod = 100,
                MinPctOverlapDutyCycles = minOctOverlapCycles,
                GlobalInhibition = false,
                NumActiveColumnsPerInhArea = 0.02 * numColumns,
                PotentialRadius = (int)(0.15 * imageSize * imageSize),
                LocalAreaDensity = -1,
                ActivationThreshold = 10,
                MaxSynapsesPerSegment = (int)(0.01 * numColumns),
                Random = new ThreadSafeRandom(42),
                StimulusThreshold = 10,
            };

            //Runnig the Experiment
            //var sp = RunExperiment(cfg, inputPrefix);
            var sp = RunExperimentWithKNNClassifier(cfg, inputPrefix);
            //Runing the Reconstruction Method Experiment
            //RunRustructuringExperiment(sp);

        }

        /// <summary>
        /// Implements the experiment.
        /// </summary>
        /// <param name="cfg"></param>
        /// <param name="inputPrefix"> The name of the images</param>
        /// <returns>The trained bersion of the SP.</returns>
        //private SpatialPooler RunExperiment(HtmConfig cfg, string inputPrefix)
        //{

        //    var mem = new Connections(cfg);

        //    bool isInStableState = false;

        //    int numColumns = 64 * 64;
        //    //Accessing the Image Folder form the Cureent Directory
        //    string trainingFolder = "Sample\\TestFiles";
        //    //Accessing the Image Folder form the Cureent Directory Foldfer
        //    var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");
        //    //Image Size
        //    int imageSize = 28;
        //    //Folder Name in the Directorty 
        //    string testName = "test_image";

        //    HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 50, (isStable, numPatterns, actColAvg, seenInputs) =>
        //    {
        //        // Event should only be fired when entering the stable state.
        //        if (isStable)
        //        {
        //            isInStableState = true;
        //            Debug.WriteLine($"Entered STABLE state: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
        //        }
        //        else
        //        {
        //            isInStableState = false;
        //            Debug.WriteLine($"INSTABLE STATE");
        //        }
        //        // Ideal SP should never enter unstable state after stable state.
        //        Debug.WriteLine($"Entered STABLE state: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
        //    }, requiredSimilarityThreshold: 0.975);

        //    // It creates the instance of Spatial Pooler Multithreaded version.
        //    SpatialPooler sp = new SpatialPooler(hpa);

        //    //Initializing the Spatial Pooler Algorithm
        //    sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

        //    //Image Size
        //    int imgSize = 28;
        //    int[] activeArray = new int[numColumns];

        //    int numStableCycles = 0;
        //    // Runnig the Traning Cycle for 5 times
        //    int maxCycles = 5;
        //    int currentCycle = 0;

        //    while (!isInStableState && currentCycle < maxCycles)
        //    {
        //        foreach (var Image in trainingImages)
        //        {
        //            //Binarizing the Images before taking Inputs for the Sp
        //            string inputBinaryImageFile = NeoCortexUtils.BinarizeImage($"{Image}", imgSize, testName);

        //            // Read Binarized and Encoded input csv file into array
        //            int[] inputVector = NeoCortexUtils.ReadCsvIntegers(inputBinaryImageFile).ToArray();

        //            int[] oldArray = new int[activeArray.Length];
        //            List<double[,]> overlapArrays = new List<double[,]>();
        //            List<double[,]> bostArrays = new List<double[,]>();

        //            sp.compute(inputVector, activeArray, true);
        //            //Getting the Active Columns
        //            var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

        //            Debug.WriteLine($"'Cycle: {currentCycle} - Image-Input: {Image}'");
        //            Debug.WriteLine($"INPUT :{Helpers.StringifyVector(inputVector)}");
        //            Debug.WriteLine($"SDR:{Helpers.StringifyVector(activeCols)}\n");
        //        }

        //        currentCycle++;

        //        // Check if the desired number of cycles is reached
        //        if (currentCycle >= maxCycles)
        //            break;

        //        // Increment numStableCycles only when it's in a stable state
        //        if (isInStableState)
        //            numStableCycles++;
        //    }

        //    return sp;
        //}

        private (SpatialPooler, HtmClassifier<string, int[]>) RunExperimentWithHTMClassifier(HtmConfig cfg, string inputPrefix)
        {
            var mem = new Connections(cfg);
            bool isInStableState = false;

            int numColumns = 64 * 64;
            string trainingFolder = "Sample\\TestFiles";
            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");
            int imgSize = 28;
            string testName = "test_image";

            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 50, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                isInStableState = isStable;
                Debug.WriteLine(isStable ? "Entered STABLE state." : "INSTABLE STATE.");
            }, requiredSimilarityThreshold: 0.975);

            SpatialPooler sp = new SpatialPooler(hpa);
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

            HtmClassifier<string, int[]> classifier = new HtmClassifier<string, int[]>();

            int[] activeArray = new int[numColumns];
            int maxCycles = 5;
            int currentCycle = 0;

            while (!isInStableState && currentCycle < maxCycles)
            {
                foreach (var image in trainingImages)
                {
                    string inputBinaryImageFile = NeoCortexUtils.BinarizeImage($"{image}", imgSize, testName);
                    int[] inputVector = NeoCortexUtils.ReadCsvIntegers(inputBinaryImageFile).ToArray();

                    sp.compute(inputVector, activeArray, true);
                    var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                    // Train the classifier: associate active columns with the image name
                    classifier.Learn(image, activeCols);

                    Debug.WriteLine($"'Cycle: {currentCycle} - Image-Input: {image}'");
                    Debug.WriteLine($"INPUT :{Helpers.StringifyVector(inputVector)}");
                    Debug.WriteLine($"SDR:{Helpers.StringifyVector(activeCols)}\n");
                }

                currentCycle++;

                if (currentCycle >= maxCycles)
                    break;
            }

            // Example prediction after training
            string testImage = trainingImages[0];
            string testBinaryImageFile = NeoCortexUtils.BinarizeImage($"{testImage}", imgSize, testName);
            int[] testInputVector = NeoCortexUtils.ReadCsvIntegers(testBinaryImageFile).ToArray();

            sp.compute(testInputVector, activeArray, false);
            var testActiveCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

            var predictions = classifier.GetPredictedInputValues(testActiveCols, 1);
            Debug.WriteLine($"Predicted label for {testImage}: {string.Join(", ", predictions.Select(p => p.PredictedInput))}");

            return (sp, classifier);
        }

        private void RunRustructuringExperiment(SpatialPooler sp)
        {
            // Path to the folder containing training images
            string trainingFolder = "Sample\\TestFiles";
            // Get all image files matching the specified prefix
            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");
            // Size of the images
            int imgSize = 28;
            // Name for the test image
            string testName = "test_image";
            // Array to hold active columns
            int[] activeArray = new int[64 * 64];
            // List to store heatmap data
            List<List<double>> heatmapData = new List<List<double>>();
            // Initialize a list to get normalized permanence values.
            List<int[]> BinarizedencodedInputs = new List<int[]>();
            // List to store normalized permanence values
            List<int[]> normalizedPermanence = new List<int[]>();
            // List to store similarity values
            List<double[]> similarityList = new List<double[]>();
            foreach (var Image in trainingImages)
            {
                string inputBinaryImageFile = NeoCortexUtils.BinarizeImage($"{Image}", imgSize, testName);

                // Read input csv file into array
                int[] inputVector = NeoCortexUtils.ReadCsvIntegers(inputBinaryImageFile).ToArray();

                // Initialize arrays and lists for computations
                int[] oldArray = new int[activeArray.Length];
                List<double[,]> overlapArrays = new List<double[,]>();
                List<double[,]> bostArrays = new List<double[,]>();

                // Compute spatial pooling on the input vector
                sp.compute(inputVector, activeArray, true);
                var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(activeCols);

                int maxInput = inputVector.Length;

                // Create a new dictionary to store extended probabilities
                Dictionary<int, double> allPermanenceDictionary = new Dictionary<int, double>();
                // Iterate through all possible inputs using a foreach loop
                foreach (var kvp in reconstructedPermanence)
                {
                    int inputIndex = kvp.Key;
                    double probability = kvp.Value;

                    // Use the existing probability
                    allPermanenceDictionary[inputIndex] = probability;
                }

                //Assinginig the inactive columns Permanence 0
                for (int inputIndex = 0; inputIndex < maxInput; inputIndex++)
                {
                    if (!reconstructedPermanence.ContainsKey(inputIndex))
                    {
                        // Key doesn't exist, set the probability to 0
                        allPermanenceDictionary[inputIndex] = 0.0;
                    }
                }

                // Sort the dictionary by keys
                var sortedAllPermanenceDictionary = allPermanenceDictionary.OrderBy(kvp => kvp.Key);
                // Convert the sorted dictionary of allpermanences to a list
                List<double> permanenceValuesList = sortedAllPermanenceDictionary.Select(kvp => kvp.Value).ToList();

                //Collecting Heatmap Data for Visualization
                heatmapData.Add(permanenceValuesList);

                //Collecting Encoded Data for Visualization
                BinarizedencodedInputs.Add(inputVector);

                //Normalizing Permanence Threshold
                var ThresholdValue = 30.5;

                // Normalize permanences (0 and 1) based on the threshold value and convert them to a list of integers.
                List<int> normalizePermanenceList = Helpers.ThresholdingProbabilities(permanenceValuesList, ThresholdValue);

                //Collecting Normalized Permanence List for Visualizing
                normalizedPermanence.Add(normalizePermanenceList.ToArray());

                //Calculating Similarity with encoded Inputs and Reconstructed Inputs
                var similarity = MathHelpers.JaccardSimilarityofBinaryArrays(inputVector, normalizePermanenceList.ToArray());

                double[] similarityArray = new double[] { similarity };

                //Collecting Similarity Data for visualizing
                similarityList.Add(similarityArray);
                Debug.WriteLine($"Similarity: {similarity}");

            }
        }



        private (SpatialPooler, KNeighborsClassifier<string, int[]>) RunExperimentWithKNNClassifier(HtmConfig cfg, string inputPrefix)
        {
            var mem = new Connections(cfg);
            bool isInStableState = false;

            int numColumns = 64 * 64;
            string trainingFolder = "Sample\\TestFiles";
            string outputFolder = "Output"; // Output folder
            Directory.CreateDirectory(outputFolder); // Ensure the output folder exists

            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");
            int imgSize = 28;
            string testName = "test_image";

            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 50, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                isInStableState = isStable;
                Debug.WriteLine(isStable ? "Entered STABLE state." : "INSTABLE STATE.");
            }, requiredSimilarityThreshold: 0.975);

            SpatialPooler sp = new SpatialPooler(hpa);
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

            KNeighborsClassifier<string, int[]> knnClassifier = new KNeighborsClassifier<string, int[]>();

            int[] activeArray = new int[numColumns];
            int maxCycles = 5;
            int currentCycle = 0;

            // Training loop
            while (!isInStableState && currentCycle < maxCycles)
            {
                foreach (var image in trainingImages)
                {
                    string inputBinaryImageFile = NeoCortexUtils.BinarizeImage($"{image}", imgSize, testName);
                    int[] inputVector = NeoCortexUtils.ReadCsvIntegers(inputBinaryImageFile).ToArray();

                    sp.compute(inputVector, activeArray, true);
                    var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                    // Convert activeCols to Cell[] format
                    var activeCells = activeCols.Select(colIdx => new Cell { Index = colIdx }).ToArray();

                    // Train the KNN classifier: associate active columns with the image name
                    knnClassifier.Learn(image, activeCells);

                    Debug.WriteLine($"'Cycle: {currentCycle} - Image-Input: {image}'");
                    Debug.WriteLine($"INPUT :{Helpers.StringifyVector(inputVector)}");
                    Debug.WriteLine($"SDR:{Helpers.StringifyVector(activeCols)}\n");

                    Debug.WriteLine($"Cycle: {currentCycle} - Image-Input: {image}");


                }

                currentCycle++;

                if (currentCycle >= maxCycles)
                    break;
            }

            // Test the classifier with the first training image (or any specific test image)
            string testImage = trainingImages[0];
            string testBinaryImageFile = NeoCortexUtils.BinarizeImage($"{testImage}", imgSize, testName);
            int[] testInputVector = NeoCortexUtils.ReadCsvIntegers(testBinaryImageFile).ToArray();

            sp.compute(testInputVector, activeArray, false);
            var testActiveCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);
            var testActiveCells = testActiveCols.Select(colIdx => new Cell { Index = colIdx }).ToArray();

            // Log the test SDR before classification
            Debug.WriteLine("\n--- TEST SDR ---");
            Debug.WriteLine($"Test Image: {testImage}");
            Debug.WriteLine($"Test SDR: {Helpers.StringifyVector(testActiveCols)}\n");

            // Print stored SDRs before comparison
            Debug.WriteLine("\n--- STORED SDRs ---");
            foreach (var label in knnClassifier.StoredSDRs.Keys)
            {
                foreach (var storedSDR in knnClassifier.StoredSDRs[label])
                {
                    Debug.WriteLine($"Label: {label}, SDR: {Helpers.StringifyVector(storedSDR)}");
                }
            }

            // Get predictions from the KNN classifier
            var predictions = knnClassifier.GetPredictedInputValues(testActiveCells, 7);

            // Show distances calculated between test and stored SDRs
            Debug.WriteLine("\n--- DISTANCE CALCULATIONS ---");
            foreach (var prediction in predictions)
            {
                Debug.WriteLine($"Compared Label: {prediction.PredictedInput}, Distance: {prediction.NumOfSameBits}");
            }

            // Display similarity scores & final predictions
            Debug.WriteLine("\n--- FINAL PREDICTIONS ---");
            foreach (var prediction in predictions)
            {
                Debug.WriteLine($"Predicted Label: {prediction.PredictedInput}, Similarity: {prediction.Similarity}");
            }

            // Optional: Limit to top predictions
            var sortedPredictions = predictions.OrderByDescending(p => p.Similarity).Take(3); // Top 3 predictions
            Debug.WriteLine("\nTop Predictions:");
            foreach (var prediction in sortedPredictions)
            {
                Debug.WriteLine($"Predicted Label: {prediction.PredictedInput}, Similarity: {prediction.Similarity}");

            }

            Debug.WriteLine("\n--- PREDICTED SDRs ---");
            foreach (var prediction in predictions)
            {
                Debug.WriteLine($"Predicted Label: {prediction.PredictedInput}");
                if (prediction.PredictedSDRs != null && prediction.PredictedSDRs.Count > 0)
                {
                    Debug.WriteLine($"Predicted SDR: {string.Join(", ", prediction.PredictedSDRs)}");
                }
                else
                {
                    Debug.WriteLine("No SDRs found for this prediction.");
                }
            }

         
            }

            // Pass the predicted SDRs to the restructuring function
            RunRustructuringExperiment2(sp, predictedSDRsList);

            return (sp, knnClassifier);


        }

        private void RunRustructuringExperiment2(SpatialPooler sp, List<int[]> predictedSDRsList)
        {
            foreach (var predictedSDR in predictedSDRsList)
            {
                // Use the predicted SDR in the reconstruction process
                Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(predictedSDR);

                int maxInput = predictedSDR.Length;

                List<List<double>> heatmapData = new List<List<double>>();
                // Initialize a list to get normalized permanence values.
                List<int[]> BinarizedencodedInputs = new List<int[]>();
                // List to store normalized permanence values
                List<int[]> normalizedPermanence = new List<int[]>();
                // List to store similarity values
                List<double[]> similarityList = new List<double[]>();

                // Create a new dictionary to store extended probabilities
                Dictionary<int, double> allPermanenceDictionary = new Dictionary<int, double>();
                // Iterate through all possible inputs using a foreach loop
                foreach (var kvp in reconstructedPermanence)
                {
                    int inputIndex = kvp.Key;
                    double probability = kvp.Value;

                    // Use the existing probability
                    allPermanenceDictionary[inputIndex] = probability;
                }

                //Assinginig the inactive columns Permanence 0
                for (int inputIndex = 0; inputIndex < maxInput; inputIndex++)
                {
                    if (!reconstructedPermanence.ContainsKey(inputIndex))
                    {
                        // Key doesn't exist, set the probability to 0
                        allPermanenceDictionary[inputIndex] = 0.0;
                    }
                }

                // Sort the dictionary by keys
                var sortedAllPermanenceDictionary = allPermanenceDictionary.OrderBy(kvp => kvp.Key);
                // Convert the sorted dictionary of allpermanences to a list
                List<double> permanenceValuesList = sortedAllPermanenceDictionary.Select(kvp => kvp.Value).ToList();

                //Collecting Heatmap Data for Visualization
                heatmapData.Add(permanenceValuesList);

                //Collecting Encoded Data for Visualization
                //BinarizedencodedInputs.Add(inputVector);

                //Normalizing Permanence Threshold
                var ThresholdValue = 25.5;

                // Normalize permanences (0 and 1) based on the threshold value and convert them to a list of integers.
                List<int> normalizePermanenceList = Helpers.ThresholdingProbabilities(permanenceValuesList, ThresholdValue);

                //Collecting Normalized Permanence List for Visualizing
                normalizedPermanence.Add(normalizePermanenceList.ToArray());
                foreach (var permanenceArray in normalizedPermanence)
                {
                    Debug.WriteLine($"[{string.Join(", ", permanenceArray)}]");
                }

                ////Calculating Similarity with encoded Inputs and Reconstructed Inputs
                //var similarity = MathHelpers.JaccardSimilarityofBinaryArrays(inputVector, normalizePermanenceList.ToArray());

                //double[] similarityArray = new double[] { similarity };

                ////Collecting Similarity Data for visualizing
                //similarityList.Add(similarityArray);
                //Debug.WriteLine($"Similarity: {similarity}");
                SaveNormalizedPermanence(normalizedPermanence, "NormalizedPermanenceOutput");

            }
        }

        private void SaveNormalizedPermanence(List<int[]> normalizedPermanence, string outputFolder)
        {
            // Ensure the output directory exists
            Directory.CreateDirectory(outputFolder);

            // Loop through each permanence array and save it
            for (int i = 0; i < normalizedPermanence.Count; i++)
            {
                // Generate a unique filename using timestamp and index
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmssfff"); // Adds millisecond precision
                string filePath = Path.Combine(outputFolder, $"normalized_{timestamp}_{i}.txt");

                using (StreamWriter writer = new StreamWriter(filePath))
                {
                    for (int row = 0; row < 32; row++)
                    {
                        string line = string.Join(" ", normalizedPermanence[i].Skip(row * 32).Take(32));
                        writer.WriteLine(line);
                    }
                }

                Debug.WriteLine($"Saved: {filePath}");
            }
        }



    }
}




//using NeoCortex;
//using NeoCortexApi.Entities;
//using NeoCortexApi.Utility;
//using NeoCortexApi;
//using System;
//using System.Collections.Generic;
//using System.Diagnostics;
//using System.IO;
//using System.Linq;
//using NeoCortexApi.Classifiers;
//using System.Text;

//namespace NeoCortexApiSample
//{
//    internal class ImageBinarizerSpatialPattern
//    {
//        public string inputPrefix { get; private set; }

//        public void Run()
//        {
//            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(ImageBinarizerSpatialPattern)}");

//            double minOctOverlapCycles = 1.0;
//            double maxBoost = 5.0;
//            int numColumns = 64 * 64;
//            int imageSize = 28;
//            var colDims = new int[] { 64, 64 };

//            HtmConfig cfg = new HtmConfig(new int[] { imageSize, imageSize }, new int[] { numColumns })
//            {
//                CellsPerColumn = 10,
//                InputDimensions = new int[] { imageSize, imageSize },
//                NumInputs = imageSize * imageSize,
//                ColumnDimensions = colDims,
//                MaxBoost = maxBoost,
//                DutyCyclePeriod = 100,
//                MinPctOverlapDutyCycles = minOctOverlapCycles,
//                GlobalInhibition = false,
//                NumActiveColumnsPerInhArea = 0.02 * numColumns,
//                PotentialRadius = (int)(0.15 * imageSize * imageSize),
//                LocalAreaDensity = -1,
//                ActivationThreshold = 10,
//                MaxSynapsesPerSegment = (int)(0.01 * numColumns),
//                Random = new ThreadSafeRandom(42),
//                StimulusThreshold = 10,
//            };

//            var sp = RunExperimentWithKNNClassifier(cfg, inputPrefix);
//        }

//        private (SpatialPooler, KNeighborsClassifier<string, int[]>) RunExperimentWithKNNClassifier(HtmConfig cfg, string inputPrefix)
//        {
//            var mem = new Connections(cfg);
//            bool isInStableState = false;

//            int numColumns = 64 * 64;
//            string trainingFolder = "Sample\\TestFiles";
//            string outputFolder = "Output";
//            Directory.CreateDirectory(outputFolder);

//            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");
//            int imgSize = 28;
//            string testName = "test_image";

//            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 50, (isStable, numPatterns, actColAvg, seenInputs) =>
//            {
//                isInStableState = isStable;
//                Debug.WriteLine(isStable ? "Entered STABLE state." : "INSTABLE STATE.");
//            }, requiredSimilarityThreshold: 0.975);

//            SpatialPooler sp = new SpatialPooler(hpa);
//            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

//            KNeighborsClassifier<string, int[]> knnClassifier = new KNeighborsClassifier<string, int[]>();

//            int[] activeArray = new int[numColumns];
//            int maxCycles = 5;
//            int currentCycle = 0;

//            Dictionary<string, List<int[]>> imageSDRs = new Dictionary<string, List<int[]>>();

//            while (!isInStableState && currentCycle < maxCycles)
//            {
//                foreach (var image in trainingImages)
//                {
//                    string inputBinaryImageFile = NeoCortexUtils.BinarizeImage(image, imgSize, testName);
//                    int[] inputVector = NeoCortexUtils.ReadCsvIntegers(inputBinaryImageFile).ToArray();
//                    Debug.WriteLine($"Binarized Input for {image}: {Helpers.StringifyVector(inputVector)}");

//                    sp.compute(inputVector, activeArray, true);
//                    var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);
//                    Debug.WriteLine($"SDR for {image} in cycle {currentCycle}: {Helpers.StringifyVector(activeCols)}");

//                    if (!imageSDRs.ContainsKey(image))
//                    {
//                        imageSDRs[image] = new List<int[]>();
//                    }
//                    imageSDRs[image].Add(activeCols.ToArray());
//                }
//                currentCycle++;
//            }

//            foreach (var image in imageSDRs)
//            {
//                var sdrs = image.Value;
//                if (sdrs.Count < 2)
//                {
//                    Debug.WriteLine($"Image: {image.Key} has less than 2 SDRs. Skipping similarity calculation.");
//                    continue;
//                }

//                double totalSimilarity = 0.0;
//                int pairCount = 0;

//                for (int i = 0; i < sdrs.Count; i++)
//                {
//                    for (int j = i + 1; j < sdrs.Count; j++)
//                    {
//                        double similarity = MathHelpers.JaccardSimilarityofBinaryArrays(sdrs[i], sdrs[j]) * 100.0;
//                        Debug.WriteLine($"SDR Similarity between instances of {image.Key}: {similarity:F2}%");
//                        totalSimilarity += similarity;
//                        pairCount++;
//                    }
//                }

//                double averageSimilarity = pairCount > 0 ? totalSimilarity / pairCount : 0.0;
//                Debug.WriteLine($"Image: {image.Key} - Average SDR Similarity: {averageSimilarity:F2}%");
//            }

//            string testImage = trainingImages[0];
//            string testBinaryImageFile = NeoCortexUtils.BinarizeImage(testImage, imgSize, testName);
//            int[] testInputVector = NeoCortexUtils.ReadCsvIntegers(testBinaryImageFile).ToArray();

//            sp.compute(testInputVector, activeArray, false);
//            var testActiveCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

//            var testActiveCells = testActiveCols.Select(colIdx => new Cell { Index = colIdx }).ToArray();
//            var predictions = knnClassifier.GetPredictedInputValues(testActiveCells, 7);

//            foreach (var prediction in predictions)
//            {
//                Debug.WriteLine($"Predicted Label: {prediction.PredictedInput}, Accuracy: {prediction.Similarity}");
//            }

//            return (sp, knnClassifier);
//        }
//    }
//}




