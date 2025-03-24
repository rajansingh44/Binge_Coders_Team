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
using System.Drawing;
using System.Drawing.Imaging;

namespace NeoCortexApiSample
{
    /// <summary>
    /// This class performs an experiment on image reconstruction using 
    /// HTM (Hierarchical Temporal Memory) and KNN (K-Nearest Neighbors) classifiers.
    /// </summary>
    internal class ImageReconstructionwithHtm_KNNClassifier
    {
        /// <summary>
        /// Gets the input prefix for processing image data.
        /// </summary>
        public string inputPrefix { get; private set; }

        /// <summary>
        /// Executes the main experiment for image reconstruction 
        /// using HTM and KNN classifiers.
        /// </summary>
        public void Run()
        {
            // Display project information
            Console.WriteLine("========================================================");
            Console.WriteLine("Project Title: ML 24/25-01 Investigate Image Reconstruction by using Classifiers");
            Console.WriteLine("Professor: Dr. Damir Dobric");
            Console.WriteLine("University: Frankfurt University Of Applied Sciences");
            Console.WriteLine("Year: 2024/25 Winter Semester");
            Console.WriteLine("Students: Rajan Singh, Pradeep Tiwari, Mausam Bhunia");
            Console.WriteLine("========================================================\n");

            // Display experiment identifier
            Console.WriteLine($"Hello People! This is the Project Experiment {nameof(ImageReconstructionwithHtm_KNNClassifier)}");

            // HTM Configuration Parameters
            double minOctOverlapCycles = 1.0; // Minimum percentage of overlap duty cycles
            double maxBoost = 5.0; // Maximum boosting factor for columns
            int numColumns = 84 * 84; // Total number of columns in spatial pooling
            int imageSize = 52; // Image size (height & width)
            var colDims = new int[] { 84, 84 }; // Column dimensions

            /// <summary>
            /// Configures the HTM network with specified parameters.
            /// </summary>
            HtmConfig cfg = new HtmConfig(new int[] { imageSize, imageSize }, new int[] { numColumns })
            {
                CellsPerColumn = 10, // Number of cells per column
                InputDimensions = new int[] { imageSize, imageSize }, // Input size
                NumInputs = imageSize * imageSize, // Total number of input features
                ColumnDimensions = colDims, // Column grid dimensions
                MaxBoost = maxBoost, // Maximum boosting value
                DutyCyclePeriod = 100, // Period over which duty cycles are calculated
                MinPctOverlapDutyCycles = minOctOverlapCycles, // Minimum overlap threshold
                NumActiveColumnsPerInhArea = 0.02 * numColumns, // Number of active columns per inhibition area
                LocalAreaDensity = -1, // Defines local inhibition settings
                MaxSynapsesPerSegment = (int)(0.01 * numColumns), // Max synapses per segment
                Random = new ThreadSafeRandom(42), // Random seed for reproducibility
                StimulusThreshold = 10, // Minimum number of active inputs required for activation
                PotentialRadius = (int)(0.5 * imageSize * imageSize), // Defines connectivity radius
                GlobalInhibition = true, // Enables global inhibition
                ActivationThreshold = 5 // Minimum synapses needed for a segment to activate
            };

            /// <summary>
            /// Runs the experiment for image reconstruction using both KNN and HTM classifiers.
            /// The function returns the spatial pooler, classifiers, predicted SDRs, and stored SDRs.
            /// </summary>
            var (sp, knnClassifier, Htmclassifier, predictedSDRsListknn, predictedSDRsListHtm, storedSDRs) =
                RunExperimentWithKNNandHTMClassifier(cfg, inputPrefix);
        }


        /// <summary>
        /// Runs the experiment using both KNN and HTM classifiers for image reconstruction.
        /// </summary>
        /// <param name="cfg">HTM configuration settings.</param>
        /// <param name="inputPrefix">Prefix for filtering input images.</param>
        /// <returns>
        /// A tuple containing:
        /// - The Spatial Pooler instance.
        /// - KNN classifier.
        /// - HTM classifier.
        /// - List of predicted SDRs from KNN.
        /// - List of predicted SDRs from HTM.
        /// - Dictionary storing SDRs mapped to image labels.
        /// </returns>
        private (SpatialPooler, KNeighborsClassifier<string, int[]>, HtmClassifier<string, int[]>, List<int[]>, List<int[]>, Dictionary<string, List<int[]>>)
            RunExperimentWithKNNandHTMClassifier(HtmConfig cfg, string inputPrefix)
        {
            // Initialize HTM memory
            var mem = new Connections(cfg);
            bool isInStableState = false;

            // Define parameters
            int numColumns = 84 * 84;
            string trainingFolder = "Sample\\TestFiles";
            string outputFolder = Path.Combine("Output");
            string sdrFolder = Path.Combine("SDRs");

            // Ensure output directories exist
            Directory.CreateDirectory(outputFolder);
            Directory.CreateDirectory(sdrFolder);

            // Load training images
            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.jpg");
            int imgSize = 52; // Fixed image size
            int maxCycles = 50; // Maximum training cycles
            int currentCycle = 0;

            Debug.WriteLine($"Initializing Training with {trainingImages.Length} images.");

            /// <summary>
            /// Initializes the Homeostatic Plasticity Controller (HPA) to regulate the stability of HTM learning.
            /// It monitors whether the model reaches a stable state based on similarity thresholds and adjusts accordingly.
            /// </summary>
            /// <param name="mem">HTM memory connections.</param>
            /// <param name="trainingImages.Length * 40">Total training iterations (determined by the number of training images).</param>
            /// <param name="callback">
            /// Callback function to check stability:
            /// - If the system is not stable, logs "INSTABLE STATE" and sets `isInStableState` to false.
            /// - If stable, logs "STABLE STATE" and sets `isInStableState` to true.
            /// </param>
            /// <param name="requiredSimilarityThreshold">Threshold (0.975) to determine when the system is considered stable.</param>
            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 40,
                 (isStable, numPatterns, actColAvg, seenInputs) =>
                 {
                     // Check if the system is stable or not
                     if (!isStable)
                     {
                         Debug.WriteLine($"INSTABLE STATE"); // Log instability
                         isInStableState = false; // Mark system as unstable
                     }
                     else
                     {
                         Debug.WriteLine($"STABLE STATE"); // Log stability
                         isInStableState = true; // Mark system as stable
                     }
                 }, requiredSimilarityThreshold: 0.975);


            // ==========================================
            //       INITIALIZATION OF HTM COMPONENTS
            // ==========================================

            /// <summary>
            /// Initializes the Spatial Pooler (SP) with Homeostatic Plasticity (HPA).
            /// The Spatial Pooler converts raw input into sparse distributed representations (SDRs).
            /// </summary>
            SpatialPooler sp = new SpatialPooler(hpa);

            /// <summary>
            /// Initializes the Spatial Pooler with memory connections.
            /// This Distributed Memory structure manages columnar representations efficiently.
            /// </summary>
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

            /// <summary>
            /// Initializes the K-Nearest Neighbors (KNN) classifier.
            /// The KNN classifier stores SDRs and uses similarity-based retrieval.
            /// </summary>
            KNeighborsClassifier<string, int[]> knnClassifier = new KNeighborsClassifier<string, int[]>();

            /// <summary>
            /// Initializes the HTM-based classifier.
            /// This classifier works similarly to KNN but is based on HTM principles.
            /// </summary>
            HtmClassifier<string, int[]> Htmclassifier = new HtmClassifier<string, int[]>();

            // ==========================================
            //       DATA STRUCTURES FOR SDR STORAGE
            // ==========================================

            /// <summary>
            /// List to store SDRs predicted using the KNN classifier.
            /// </summary>
            List<int[]> predictedSDRsListKnn = new List<int[]>();

            /// <summary>
            /// List to store SDRs predicted using the HTM classifier.
            /// </summary>
            List<int[]> predictedSDRsListHtm = new List<int[]>();

            /// <summary>
            /// Dictionary to store labeled SDRs.
            /// The key represents the image label (file name), and the value is a list of SDRs associated with that label.
            /// </summary>
            Dictionary<string, List<int[]>> storedSDRs = new Dictionary<string, List<int[]>>();

            /// <summary>
            /// Array to store active columns identified by the Spatial Pooler.
            /// </summary>
            int[] activeArray = new int[numColumns];

            // ==========================================
            //        TRAINING PHASE: PROCESS IMAGES
            // ==========================================

            /// <summary>
            /// This loop runs multiple training cycles until the system reaches a stable state.
            /// It processes images, generates SDRs, and trains both KNN and HTM classifiers.
            /// </summary>
            while (!isInStableState && currentCycle < maxCycles)
            {
                Debug.WriteLine($"\n Training Cycle {currentCycle + 1}/{maxCycles} ");

                foreach (var image in trainingImages)
                {
                    try
                    {
                        Debug.WriteLine($" Processing Image: {image}");

                        // Step 1: Convert image into a binary format.
                        string binarizedImageFile = BinarizeImageToFixedSize(image, imgSize);
                        Debug.WriteLine($" Binarized Image File: {binarizedImageFile}");

                        // Step 2: Read the binary representation of the image.
                        int[] inputVector = ReadBinaryTextFile(binarizedImageFile);
                        Debug.WriteLine($" Input Vector Length: {inputVector.Length}");

                        // Step 3: Compute active columns using the Spatial Pooler.
                        sp.compute(inputVector, activeArray, true);
                        var activeCols = ArrayUtils.IndexWhere(activeArray, el => el == 1);
                        Debug.WriteLine($" Active Columns Count: {activeCols.Length}");

                        // Step 4: Convert active columns into SDR format.
                        var activeCells = activeCols.Select(colIdx => new NeoCortexApi.Entities.Cell { Index = colIdx }).ToArray();

                        // Step 5: Train the KNN classifier.
                        knnClassifier.Learn(image, activeCells);
                        Debug.WriteLine($" KNN Learning from {image}, Stored SDR: {string.Join(",", activeCols)}");

                        // Step 6: Train the HTM classifier.
                        Htmclassifier.Learn(image, activeCols);
                        Debug.WriteLine($" HTM Learning from {image}, Stored SDR: {string.Join(",", activeCols)}");

                        // Step 7: Store SDR in the dictionary for later retrieval.
                        if (!storedSDRs.ContainsKey(image))
                        {
                            storedSDRs[image] = new List<int[]>();
                        }
                        storedSDRs[image].Add(activeCols);

                        // Step 8: Store SDR as a file in the SDR folder.
                        predictedSDRsListKnn.Add(activeCols);
                        predictedSDRsListHtm.Add(activeCols);
                        string sdrFilePath = Path.Combine(sdrFolder, $"{Path.GetFileNameWithoutExtension(image)}.csv");
                        File.WriteAllText(sdrFilePath, string.Join(",", activeCols));
                        Debug.WriteLine($" Stored SDR for {image} at {sdrFilePath}");
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($" Error processing {image}: {ex.Message}");
                    }
                }

                // Increment the training cycle count.
                currentCycle++;
                Debug.WriteLine($" Completed Cycle {currentCycle}.");
            }

            /// <summary>
            /// Checks whether the model has reached a stable state after training.
            /// </summary>
            if (!isInStableState)
            {
                Debug.WriteLine(" Training completed, but stable state not reached.");
            }
            else
            {
                Debug.WriteLine(" Training completed successfully.");
            }

            // ==========================================
            //     LOG PREDICTED SDRs FOR CLASSIFICATION
            // ==========================================

            /// <summary>
            /// Logs all SDRs predicted using the KNN classifier.
            /// </summary>
            Debug.WriteLine("\n--- PREDICTED KNN SDRs ---");
            foreach (var sdr in predictedSDRsListKnn)
            {
                Debug.WriteLine($"Predicted SDR: {string.Join(", ", sdr)}");
            }

            /// <summary>
            /// Logs all SDRs predicted using the HTM classifier.
            /// </summary>
            Debug.WriteLine("\n--- PREDICTED HTM SDRs ---");
            foreach (var sdr in predictedSDRsListHtm)
            {
                Debug.WriteLine($"Predicted SDR: {string.Join(", ", sdr)}");
            }

            /// <summary>
            / Logs all stored SDRs before classification.
            / </ summary >
            Debug.WriteLine("\n--- STORED SDRs ---");
            foreach (var label in storedSDRs.Keys)
            {
                foreach (var storedSDR in storedSDRs[label])
                {
                    Debug.WriteLine($"Label: {label}, SDR: {string.Join(", ", storedSDR)}");
                }
            }

            // ==========================================
            //           CLASSIFICATION PHASE
            // ==========================================

            /// <summary>
            /// Performs KNN-based classification for predicted SDRs.
            /// </summary>
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

            /// <summary>
            /// Performs HTM-based classification for predicted SDRs.
            /// </summary>
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

            // ==========================================
            //         IMAGE RECONSTRUCTION PHASE
            // ==========================================

            /// <summary>
            /// Runs image reconstruction experiment using KNN classifier's SDRs.
            /// </summary>
            RunRustructuringExperimentKNN(sp, predictedSDRsListKnn);
            Debug.WriteLine("\n Running KNN Restructuring Experiment...");

            /// <summary>
            /// Runs image reconstruction experiment using HTM classifier's SDRs.
            /// </summary>
            RunRustructuringExperimentHtm(sp, predictedSDRsListHtm);
            Debug.WriteLine("\n Running HTM Restructuring Experiment...");

            // ==========================================
            //     RETURN TRAINED COMPONENTS & RESULTS
            // ==========================================

            /// <summary>
            /// Returns trained classifiers, predicted SDRs, and stored SDRs for further processing.
            /// </summary>
            return (sp, knnClassifier, Htmclassifier, predictedSDRsListKnn, predictedSDRsListHtm, storedSDRs);
        }

        /// <summary>
        /// Reconstructs images from predicted SDRs.
        /// </summary>
        private void RunRustructuringExperimentKNN(SpatialPooler sp, List<int[]> predictedSDRsList)
        {
            List<int[]> normalizedPermanence = new List<int[]>(); // List to store normalized permanence values
            List<string> cosineResults = new List<string>(); // List to store cosine similarity results

            foreach (var predictedSDR in predictedSDRsList)
            {
                Debug.WriteLine("Reconstructing permanence for SDR...");

                // Reconstruct the permanence values for the predicted SDR
                Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(predictedSDR);

                Dictionary<int, double> allPermanenceDictionary = new Dictionary<int, double>();
                foreach (var kvp in reconstructedPermanence)
                {
                    allPermanenceDictionary[kvp.Key] = kvp.Value;
                }

                int imgsize = 52 * 52; // Assuming image size is 52x52 pixels

                // Assign inactive columns a permanence value of 0
                for (int inputIndex = 0; inputIndex < imgsize; inputIndex++)
                {
                    if (!reconstructedPermanence.ContainsKey(inputIndex))
                    {
                        allPermanenceDictionary[inputIndex] = 0.0;
                    }
                }

                // Normalize permanence values using a threshold
                var ThresholdValue = 70.0;
                List<double> permanenceValuesList = allPermanenceDictionary.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();
                List<int> normalizePermanenceList = Helpers.ThresholdingforResetImg(permanenceValuesList, ThresholdValue);

                normalizedPermanence.Add(normalizePermanenceList.ToArray());

                // Save the reconstructed binary image
                string outputPath = $"ReconstructedSDR_{predictedSDRsList.IndexOf(predictedSDR)}";
                NeoCortexUtils.SaveBinarizedImageFromBinaryArray(normalizePermanenceList.ToArray(), outputPath);
                Debug.WriteLine($"Reconstructed Image saved at {outputPath}");

                // Calculate Cosine Similarity between original SDR and reconstructed SDR
                double similarity = CosineSimilarity(predictedSDR, normalizePermanenceList.ToArray());
                double similarityPercentage = similarity * 100;
                cosineResults.Add($"{outputPath},{similarityPercentage:F2}");

                Debug.WriteLine($"KNN Similarity between {outputPath} and original KNN SDR: {similarityPercentage:F2}%");
            }

            // Save Cosine Similarity results to CSV
            string Cosine = "KNN_Similarity_Results";
            Directory.CreateDirectory(Cosine);
            File.WriteAllLines(Path.Combine(Cosine, "Similarity_KNN.csv"), cosineResults);
        }

        /// <summary>
        /// Calculates the Cosine Similarity between two binary vectors.
        /// </summary>
        private double CosineSimilarity(int[] vec1, int[] vec2)
        {
            double dotProduct = 0, magnitude1 = 0, magnitude2 = 0;

            for (int i = 0; i < vec1.Length; i++)
            {
                dotProduct += vec1[i] * vec2[i];
                magnitude1 += vec1[i] * vec1[i];
                magnitude2 += vec2[i] * vec2[i];
            }

            return magnitude1 == 0 || magnitude2 == 0 ? 0 : dotProduct / (Math.Sqrt(magnitude1) * Math.Sqrt(magnitude2));
        }

        public void RunRustructuringExperimentHtm(SpatialPooler sp, List<int[]> predictedSDRsList)
        {
            List<int[]> normalizedPermanence_a = new List<int[]>(); // List to store normalized permanence values
            List<double[]> similarityList = new List<double[]>(); // List for storing similarity values for graph plotting
            List<string> jaccardResults = new List<string>(); // List to store Jaccard similarity results

            foreach (var predictedSDR in predictedSDRsList)
            {
                Debug.WriteLine("Reconstructing permanence for SDR...");

                // Reconstruct the permanence values for the predicted SDR
                Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(predictedSDR);
                Dictionary<int, double> allPermanenceDictionary = new Dictionary<int, double>();

                foreach (var kvp in reconstructedPermanence)
                {
                    allPermanenceDictionary[kvp.Key] = kvp.Value;
                }

                int imgsize = 52 * 52; // Assuming image size is 52x52 pixels

                // Assign inactive columns a permanence value of 0
                for (int inputIndex = 0; inputIndex < imgsize; inputIndex++)
                {
                    if (!reconstructedPermanence.ContainsKey(inputIndex))
                    {
                        allPermanenceDictionary[inputIndex] = 0.0;
                    }
                }

                // Normalize permanence values using a threshold
                var ThresholdValue = 67.0;
                List<double> permanenceValuesList = allPermanenceDictionary.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();
                List<int> normalizePermanenceList = Helpers.ThresholdingforResetImg(permanenceValuesList, ThresholdValue);
                normalizedPermanence_a.Add(normalizePermanenceList.ToArray());

                // Save the reconstructed binary image
                string outputPath = $"ReconstructedSDR_{predictedSDRsList.IndexOf(predictedSDR)}";
                NeoCortexUtils.SaveBinarizedImageFromBinaryArray_HTM(normalizePermanenceList.ToArray(), outputPath);
                Debug.WriteLine($"Reconstructed Image saved at {outputPath}");

                // Calculate Similarity between original SDR and reconstructed SDR
                double jaccardSimilarity = AdjustedJaccardSimilarity(predictedSDR, normalizePermanenceList.ToArray());
                double similarityPercentage = jaccardSimilarity * 100;
                jaccardResults.Add($"{outputPath},{similarityPercentage:F2}");
                Debug.WriteLine($"Similarity between {outputPath} and original HTM SDR: {similarityPercentage:F2}%");

                int[] inputVector = normalizePermanenceList.ToArray();

                // Prepare data for similarity graph plotting
                int[] sortedPredictedSDR = predictedSDR.OrderByDescending(x => x).ToArray();
                int[] sortedNormalizePermanenceList = normalizePermanenceList.ToArray().OrderByDescending(x => x).ToArray();

                // Collect similarity data for visualization
                double[] similarityArray = new double[] { similarityPercentage };
                similarityList.Add(similarityArray);
            }

            // Generate the similarity graph using collected data
            DrawSimilarityPlots(similarityList);

            // Save Jaccard Similarity results to CSV
            string jaccardDir = "HTM_Similarity_Results";
            Directory.CreateDirectory(jaccardDir);
            File.WriteAllLines(Path.Combine(jaccardDir, "Similarity_HTM.csv"), jaccardResults);
            CreateCombinedSimilarityCSV();
        }

        /// <summary>
        /// Calculates the Jaccard Similarity between two binary vectors.
        /// </summary>
        private double AdjustedJaccardSimilarity(int[] vec1, int[] vec2)
        {
            double dotProduct = 0, magnitude1 = 0, magnitude2 = 0;

            for (int i = 0; i < vec1.Length; i++)
            {
                dotProduct += vec1[i] * vec2[i];
                magnitude1 += vec1[i] * vec1[i];
                magnitude2 += vec2[i] * vec2[i];
            }

            return magnitude1 == 0 || magnitude2 == 0 ? 0 : dotProduct / (Math.Sqrt(magnitude1) * Math.Sqrt(magnitude2));
        }


        /// <summary>
        /// Creates a combined similarity CSV file from KNN and HTM similarity results.
        /// Generates bar charts divided into 6 images if there are more than 350 values.
        /// </summary>
        private void CreateCombinedSimilarityCSV()
        {
            string knnFilePath = Path.Combine("KNN_Similarity_Results", "Similarity_KNN.csv");
            string htmFilePath = Path.Combine("JaccardSimilarityResults", "Similarity_HTM.csv");
            string combinedDir = "CombinedSimilarityResults";
            string combinedFilePath = Path.Combine(combinedDir, "Similarity_Combined.csv");

            Directory.CreateDirectory(combinedDir);

            List<string> knnLines = File.Exists(knnFilePath) ? File.ReadAllLines(knnFilePath).ToList() : new List<string>();
            List<string> htmLines = File.Exists(htmFilePath) ? File.ReadAllLines(htmFilePath).ToList() : new List<string>();

            List<string> combinedResults = new List<string> { "Image, KNN Similarity (%), HTM Similarity (%)" };
            List<string> imageNames = new List<string>();
            List<double> knnSimilarities = new List<double>();
            List<double> htmSimilarities = new List<double>();

            int maxLines = Math.Max(knnLines.Count, htmLines.Count);

            for (int i = 0; i < maxLines; i++)
            {
                string knnEntry = i < knnLines.Count ? knnLines[i] : "N/A, N/A";
                string htmEntry = i < htmLines.Count ? htmLines[i].Split(',')[1] : "N/A";

                string imageName = knnEntry.Split(',')[0];
                string knnSimilarity = knnEntry.Split(',').Length > 1 ? knnEntry.Split(',')[1] : "N/A";

                combinedResults.Add($"{imageName}, {knnSimilarity}, {htmEntry}");

                if (double.TryParse(knnSimilarity, out double knnValue) && double.TryParse(htmEntry, out double htmValue))
                {
                    imageNames.Add(imageName);
                    knnSimilarities.Add(knnValue);
                    htmSimilarities.Add(htmValue);
                }
            }

            File.WriteAllLines(combinedFilePath, combinedResults);
            Debug.WriteLine("Combined similarity CSV generated successfully.");

            // Generate similarity comparison graphs
            GenerateSimilarityGraph(imageNames, knnSimilarities, htmSimilarities, combinedDir);
        }


        /// <summary>
        /// Generates similarity graphs comparing KNN and HTM similarity percentages as bar charts.
        /// </summary>
        /// <param name="imageNames">List of image names corresponding to similarity values.</param>
        /// <param name="knnSimilarities">List of similarity percentages computed using KNN.</param>
        /// <param name="htmSimilarities">List of similarity percentages computed using HTM.</param>
        /// <param name="saveDir">Directory where the generated graphs should be stored.</param>
        private static void GenerateSimilarityGraph(List<string> imageNames, List<double> knnSimilarities, List<double> htmSimilarities, string saveDir)
        {
            // Define graph properties
            int width = 800;
            int height = 600;
            int padding = 80;
            int graphWidth = width - 2 * padding;
            int graphHeight = height - 2 * padding;

            int numPoints = Math.Min(knnSimilarities.Count, htmSimilarities.Count);
            if (numPoints == 0) return;

            int batchSize = (int)Math.Ceiling(numPoints / 6.0);  // Split into 6 images

            for (int i = 0; i < 6; i++)
            {
                int startIdx = i * batchSize;
                int endIdx = Math.Min(startIdx + batchSize, numPoints);

                if (startIdx >= endIdx) break; // Avoid empty graphs

                using (Bitmap bitmap = new Bitmap(width, height))
                using (Graphics g = Graphics.FromImage(bitmap))
                {
                    g.Clear(Color.White);

                    // Define fonts and pens
                    Font axisFont = new Font(FontFamily.GenericSansSerif, 10, FontStyle.Bold);
                    Font titleFont = new Font(FontFamily.GenericSansSerif, 14, FontStyle.Bold);
                    Brush knnBrush = Brushes.Blue;
                    Brush htmBrush = Brushes.Red;
                    Pen axisPen = new Pen(Color.Black, 2);

                    // Draw axes
                    g.DrawLine(axisPen, padding, height - padding, padding, padding); // Y-axis
                    g.DrawLine(axisPen, padding, height - padding, width - padding, height - padding); // X-axis

                    // Labels
                    g.DrawString("Similarity %", axisFont, Brushes.Black, 10, (height / 2) - 20, new StringFormat { FormatFlags = StringFormatFlags.DirectionVertical });
                    g.DrawString($"Image Index {startIdx} - {endIdx}", axisFont, Brushes.Black, width / 3, height - 40);

                    // Y-axis scale (0% to 100%)
                    for (int y = 0; y <= 100; y += 20)
                    {
                        int yPos = height - padding - (int)(y / 100.0 * graphHeight);
                        g.DrawString($"{y}%", axisFont, Brushes.Black, padding - 40, yPos - 5);
                        g.DrawLine(Pens.Gray, padding - 5, yPos, padding + graphWidth, yPos);
                    }

                    // X-axis scale
                    int barWidth = Math.Max(5, graphWidth / (endIdx - startIdx));
                    for (int j = startIdx; j < endIdx; j++)
                    {
                        int xPos = padding + (j - startIdx) * barWidth;
                        if ((j - startIdx) % 10 == 0)  // Show every 10th index
                        {
                            g.DrawString(j.ToString(), axisFont, Brushes.Black, xPos, height - padding + 10);
                        }

                        // Draw bars
                        int knnHeight = (int)(knnSimilarities[j] / 100.0 * graphHeight);
                        int htmHeight = (int)(htmSimilarities[j] / 100.0 * graphHeight);

                        g.FillRectangle(knnBrush, xPos, height - padding - knnHeight, barWidth / 2, knnHeight);
                        g.FillRectangle(htmBrush, xPos + barWidth / 2, height - padding - htmHeight, barWidth / 2, htmHeight);
                    }

                    // Legend
                    g.FillRectangle(Brushes.White, width - 180, padding - 10, 160, 60);
                    g.DrawRectangle(Pens.Black, width - 180, padding - 10, 160, 60);
                    g.DrawString("Legend:", titleFont, Brushes.Black, width - 170, padding);
                    g.DrawString("KNN Similarity", axisFont, Brushes.Blue, width - 170, padding + 20);
                    g.DrawString("HTM Similarity", axisFont, Brushes.Red, width - 170, padding + 40);

                    // Save graph
                    string imagePath = Path.Combine(saveDir, $"SimilarityGraph_{i + 1}.png");
                    bitmap.Save(imagePath, ImageFormat.Png);
                }
            }

            Debug.WriteLine("Generated bar chart similarity graphs successfully.");
        }


        public static void DrawSimilarityPlots(List<double[]> similaritiesList)
        {
            // Combine all similarities from the list of arrays

            List<double> combinedSimilarities = new List<double>();
            foreach (var similarities in similaritiesList)

            {
                combinedSimilarities.AddRange(similarities);
            }

            // Define the folder path based on the current directory

            string folderPath = Path.Combine(Environment.CurrentDirectory, "SimilarityPlots_Image_Inputs");


            // Create the folder if it doesn't exist

            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            // Define the file name
            string fileName = "combined_similarity_plot_Image_Inputs.png";

            // Define the file path with the folder path and file name

            string filePath = Path.Combine(folderPath, fileName);

            // Draw the combined similarity plot
            NeoCortexUtils.DrawCombinedSimilarityPlot(combinedSimilarities, filePath, 2000, 2000);

            Debug.WriteLine($"Combined similarity plot generated and saved successfully.");

        }

        private static string BinarizeImageToFixedSize(string imagePath, int gridSize)
        {
            string outputFile = Path.Combine("Output", Path.GetFileNameWithoutExtension(imagePath) + ".txt");

            using (Bitmap originalImage = new Bitmap(imagePath))
            using (Bitmap resizedImage = new Bitmap(originalImage, new Size(gridSize, gridSize)))
            {
                int[] binaryArray = new int[gridSize * gridSize];

                for (int y = 0; y < gridSize; y++)
                {
                    for (int x = 0; x < gridSize; x++)
                    {
                        Color pixelColor = resizedImage.GetPixel(x, y);
                        int grayValue = (pixelColor.R + pixelColor.G + pixelColor.B) / 3;
                        binaryArray[y * gridSize + x] = (grayValue > 128) ? 1 : 0;
                    }
                }

                using (StreamWriter writer = new StreamWriter(outputFile))
                {
                    for (int i = 0; i < gridSize; i++)
                    {
                        string row = new string(binaryArray.Skip(i * gridSize).Take(gridSize).ToArray());
                        writer.WriteLine(row);
                    }
                }
                   

            }

            return outputFile;
        }


        private int[] ReadBinaryTextFile(string filePath)
        {
            var lines = File.ReadAllLines(filePath);
            return lines.SelectMany(line => line.Select(c => c == '1' ? 1 : 0)).ToArray();
        }


    }
}
