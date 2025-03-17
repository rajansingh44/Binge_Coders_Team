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
    internal class ImageReconstructionwithHtm_KNNClassifier
    {
        public string inputPrefix { get; private set; }

        public void Run()
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(ImageReconstructionwithHtm_KNNClassifier)}");

            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            int numColumns = 84 * 84;
            int imageSize = 52;
            var colDims = new int[] { 84, 84 };

            HtmConfig cfg = new HtmConfig(new int[] { imageSize, imageSize }, new int[] { numColumns })
            {
                CellsPerColumn = 10,
                InputDimensions = new int[] { imageSize, imageSize },
                NumInputs = imageSize * imageSize,
                ColumnDimensions = colDims,
                MaxBoost = maxBoost,
                DutyCyclePeriod = 100,
                MinPctOverlapDutyCycles = minOctOverlapCycles,
                NumActiveColumnsPerInhArea = 0.02 * numColumns,
                LocalAreaDensity = -1,
                MaxSynapsesPerSegment = (int)(0.01 * numColumns),
                Random = new ThreadSafeRandom(42),
                StimulusThreshold = 10,
                PotentialRadius = (int)(0.5 * imageSize * imageSize),
                GlobalInhibition = true,
                ActivationThreshold = 5
            };

            // Correct the deconstruction to handle 6 elements
            var (sp, knnClassifier, Htmclassifier, predictedSDRsListknn, predictedSDRsListHtm, storedSDRs) = RunExperimentWithKNNandHTMClassifier(cfg, inputPrefix);

        }

        private (SpatialPooler, KNeighborsClassifier<string, int[]>, HtmClassifier<string, int[]>, List<int[]>, List<int[]>, Dictionary<string, List<int[]>>) RunExperimentWithKNNandHTMClassifier(HtmConfig cfg, string inputPrefix)
        {
            var mem = new Connections(cfg);
            bool isInStableState = false;

            int numColumns = 84 * 84;
            string trainingFolder = "Sample\\TestFiles";
            string outputFolder = Path.Combine("Output");
            string sdrFolder = Path.Combine("SDRs");

            Directory.CreateDirectory(outputFolder);
            Directory.CreateDirectory(sdrFolder);

            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.jpg");
            int imgSize = 52;
            string testName = "test_image";

            Debug.WriteLine($"Initializing Training with {trainingImages.Length} images.");

            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 40,
                 (isStable, numPatterns, actColAvg, seenInputs) =>
                 {
                     if (isStable == false)
                     {
                         Debug.WriteLine($"INSTABLE STATE");
                         isInStableState = false;
                     }
                     else
                     {
                         Debug.WriteLine($"STABLE STATE");
                         isInStableState = true;
                     }
                 }, requiredSimilarityThreshold: 0.975);

            SpatialPooler sp = new SpatialPooler(hpa);
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

            KNeighborsClassifier<string, int[]> knnClassifier = new KNeighborsClassifier<string, int[]>();

            HtmClassifier<string, int[]> Htmclassifier = new HtmClassifier<string, int[]>();

            List<int[]> predictedSDRsListKnn = new List<int[]>();
            List<int[]> predictedSDRsListHtm = new List<int[]>();

            Dictionary<string, List<int[]>> storedSDRs = new Dictionary<string, List<int[]>>();

            int[] activeArray = new int[numColumns];
            int maxCycles = 50;
            int currentCycle = 0;

            // Training Phase
            while (!isInStableState && currentCycle < maxCycles)
            {
                Debug.WriteLine($"\n Training Cycle {currentCycle + 1}/{maxCycles} ");

                foreach (var image in trainingImages)
                {
                    try
                    {
                        Debug.WriteLine($" Processing Image: {image}");

                        // Binarize Image
                        string binarizedImageFile = BinarizeImageToFixedSize(image, imgSize);
                        Debug.WriteLine($" Binarized Image File: {binarizedImageFile}");

                        // Read Input Vector
                        int[] inputVector = ReadBinaryTextFile(binarizedImageFile);
                        Debug.WriteLine($" Input Vector Length: {inputVector.Length}");

                        // Compute Active Columns
                        sp.compute(inputVector, activeArray, true);
                        var activeCols = ArrayUtils.IndexWhere(activeArray, el => el == 1);
                        Debug.WriteLine($" Active Columns Count: {activeCols.Length}");

                        // Train KNN
                        var activeCells = activeCols.Select(colIdx => new NeoCortexApi.Entities.Cell { Index = colIdx }).ToArray();
                        knnClassifier.Learn(image, activeCells);


                        Debug.WriteLine($" KNN Learning from {image}, Stored SDR: {string.Join(",", activeCols)}");

                        Htmclassifier.Learn(image, activeCols);
                        Debug.WriteLine($" HTM Learning from {image}, Stored SDR: {string.Join(",", activeCols)}");

                        // Store SDR manually in the dictionary
                        if (!storedSDRs.ContainsKey(image))
                        {
                            storedSDRs[image] = new List<int[]>();
                        }
                        storedSDRs[image].Add(activeCols);  // Add the SDR to the dictionary

                        // Store SDR to file
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

                currentCycle++;
                Debug.WriteLine($" Completed Cycle {currentCycle}.");
            }

            if (!isInStableState)
            {
                Debug.WriteLine(" Training completed, but stable state not reached.");
            }
            else
            {
                Debug.WriteLine(" Training completed successfully.");
            }

            // Log and Pass Predicted SDRs
            Debug.WriteLine("\n--- PREDICTED KNN SDRs ---");
            foreach (var sdr in predictedSDRsListKnn)
            {
                Debug.WriteLine($"Predicted SDR: {string.Join(", ", sdr)}");
            }

            Debug.WriteLine("\n--- PREDICTED HTM SDRs ---");
            foreach (var sdr in predictedSDRsListHtm)
            {
                Debug.WriteLine($"Predicted SDR: {string.Join(", ", sdr)}");
            }

            // Log Stored SDRs Before Classification
            Debug.WriteLine("\n--- STORED SDRs ---");
            foreach (var label in storedSDRs.Keys)
            {
                foreach (var storedSDR in storedSDRs[label])
                {
                    Debug.WriteLine($"Label: {label}, SDR: {string.Join(", ", storedSDR)}");
                }
            }

            // Classification & Similarity Scores
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

            // Pass the predicted SDRs to the restructuring function
            RunRustructuringExperiment2(sp, predictedSDRsListKnn);
            Debug.WriteLine("\n Running KNN Restructuring Experiment...");

            RunRustructuringExperimentHtm(sp, predictedSDRsListHtm);
            Debug.WriteLine("\n Running HTM Restructuring Experiment...");

            return (sp, knnClassifier, Htmclassifier, predictedSDRsListKnn, predictedSDRsListHtm, storedSDRs);
        }

        /// <summary>
        /// Reconstructs images from predicted SDRs.
        /// </summary>
        private void RunRustructuringExperiment2(SpatialPooler sp, List<int[]> predictedSDRsList)
        {
            List<int[]> normalizedPermanence = new List<int[]>();
            List<string> cosineResults = new List<string>();

            foreach (var predictedSDR in predictedSDRsList)
            {
                Debug.WriteLine("Reconstructing permanence for SDR...");

                // Reconstruct the permanence for the predicted SDR
                Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(predictedSDR);

                Dictionary<int, double> allPermanenceDictionary = new Dictionary<int, double>();
                foreach (var kvp in reconstructedPermanence)
                {
                    allPermanenceDictionary[kvp.Key] = kvp.Value;
                }

                int imgsize = 52 * 52;

                // Assign inactive columns permanence 0
                for (int inputIndex = 0; inputIndex < imgsize; inputIndex++)
                {
                    if (!reconstructedPermanence.ContainsKey(inputIndex))
                    {
                        allPermanenceDictionary[inputIndex] = 0.0;
                    }
                }

                // Normalize permanence values
                var ThresholdValue = 67.0;
                List<double> permanenceValuesList = allPermanenceDictionary.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();
                List<int> normalizePermanenceList = Helpers.ThresholdingforResetImg(permanenceValuesList, ThresholdValue);

                normalizedPermanence.Add(normalizePermanenceList.ToArray());

                // Save the reconstructed binary image
                string outputPath = $"ReconstructedSDR_{predictedSDRsList.IndexOf(predictedSDR)}";
                NeoCortexUtils.SaveBinarizedImageFromBinaryArray(normalizePermanenceList.ToArray(), outputPath);
                Debug.WriteLine($"Reconstructed Image saved at {outputPath}");

                // *Calculate and Print Cosine Similarity*
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



        private void RunRustructuringExperimentHtm(SpatialPooler sp, List<int[]> predictedSDRsList)
        {
            List<int[]> normalizedPermanence = new List<int[]>();
            List<double[]> similarityList = new List<double[]>();
            List<string> jaccardResults = new List<string>();


            foreach (var predictedSDR in predictedSDRsList)
            {
                Debug.WriteLine("Reconstructing permanence for SDR...");

                // Reconstruct the permanence for the predicted SDR
                Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(predictedSDR);
                Dictionary<int, double> allPermanenceDictionary = new Dictionary<int, double>();

                foreach (var kvp in reconstructedPermanence)
                {
                    allPermanenceDictionary[kvp.Key] = kvp.Value;
                }

                int imgsize = 52 * 52;

                // Assign inactive columns permanence 0
                for (int inputIndex = 0; inputIndex < imgsize; inputIndex++)
                {
                    if (!reconstructedPermanence.ContainsKey(inputIndex))
                    {
                        allPermanenceDictionary[inputIndex] = 0.0;
                    }
                }

                // Normalize permanence values
                var ThresholdValue = 70.0;
                List<double> permanenceValuesList = allPermanenceDictionary.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();
                List<int> normalizePermanenceList = Helpers.ThresholdingforResetImg(permanenceValuesList, ThresholdValue);
                normalizedPermanence.Add(normalizePermanenceList.ToArray());

                // Save the reconstructed binary image
                string outputPath = $"ReconstructedSDR_{predictedSDRsList.IndexOf(predictedSDR)}";
                NeoCortexUtils.SaveBinarizedImageFromBinaryArray_HTM(normalizePermanenceList.ToArray(), outputPath);
                Debug.WriteLine($"Reconstructed Image saved at {outputPath}");

                //print the SDR and Permanance Values
                double jaccardSimilarity = JaccardSimilarity(predictedSDR, normalizePermanenceList.ToArray());
                double similarityPercentage = jaccardSimilarity * 100;
                jaccardResults.Add($"{outputPath},{similarityPercentage:F2}");
                Debug.WriteLine($"Similarity between {outputPath} and original HTM SDR: {similarityPercentage:F2}%");

                int[] inputVector = normalizePermanenceList.ToArray();


                //Calculating Similarity with encoded Inputs and Reconstructed Inputs
                var similarity = MathHelpers.JaccardSimilarityofBinaryArrays(inputVector, normalizePermanenceList.ToArray());



                double[] similarityArray = new double[] { similarity };

                //Collecting Similarity Data for visualizing
                similarityList.Add(similarityArray);
            }
            // Generate the Similarity graph using the Similarity list
            DrawSimilarityPlots(similarityList);
            // Save Jaccard Similarity results to CSV
            string jaccardDir = "JaccardSimilarityResults";
            Directory.CreateDirectory(jaccardDir);
            File.WriteAllLines(Path.Combine(jaccardDir, "Similarity_HTM.csv"), jaccardResults);
            CreateCombinedSimilarityCSV();
        }

        private double JaccardSimilarity(int[] vec1, int[] vec2)
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

        private void CreateCombinedSimilarityCSV()
        {
            string knnFilePath = Path.Combine("KNN_Similarity_Results", "Similarity_KNN.csv");
            string htmFilePath = Path.Combine("JaccardSimilarityResults", "Similarity_HTM.csv");
            string combinedFilePath = Path.Combine("CombinedSimilarityResults", "Similarity_Combined.csv");

            Directory.CreateDirectory("CombinedSimilarityResults");

            List<string> knnLines = File.Exists(knnFilePath) ? File.ReadAllLines(knnFilePath).ToList() : new List<string>();
            List<string> htmLines = File.Exists(htmFilePath) ? File.ReadAllLines(htmFilePath).ToList() : new List<string>();

            List<string> combinedResults = new List<string> { "Image, KNN Similarity (%), HTM Similarity (%)" };

            int maxLines = Math.Max(knnLines.Count, htmLines.Count);

            for (int i = 0; i < maxLines; i++)
            {
                string knnEntry = i < knnLines.Count ? knnLines[i] : "N/A, N/A";
                string htmEntry = i < htmLines.Count ? htmLines[i].Split(',')[1] : "N/A";

                string imageName = knnEntry.Split(',')[0];
                string knnSimilarity = knnEntry.Split(',').Length > 1 ? knnEntry.Split(',')[1] : "N/A";

                combinedResults.Add($"{imageName}, {knnSimilarity}, {htmEntry}");
            }

            File.WriteAllLines(combinedFilePath, combinedResults);
            Debug.WriteLine("Combined similarity CSV generated successfully.");
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





        private string BinarizeImageToFixedSize(string imagePath, int gridSize)
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
                        writer.WriteLine(string.Join("", binaryArray.Skip(i * gridSize).Take(gridSize)));
                    }
                }
            }

            return outputFile;
        }
    }
}