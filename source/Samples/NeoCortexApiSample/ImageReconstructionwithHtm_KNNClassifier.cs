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

            var (sp, knnClassifier, predictedSDRsList) = RunExperimentWithKNNClassifier(cfg, inputPrefix);

            // Run the Reconstruction Experiment
            RunRustructuringExperiment2(sp, predictedSDRsList);
        }

        private (SpatialPooler, KNeighborsClassifier<string, int[]>, List<int[]>) RunExperimentWithKNNClassifier(HtmConfig cfg, string inputPrefix)
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

            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 50, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                isInStableState = isStable;
                Debug.WriteLine(isInStableState ? "🚀 Entered STABLE state." : "⚠ INSTABLE STATE.");
            }, requiredSimilarityThreshold: 0.975);

            SpatialPooler sp = new SpatialPooler(hpa);
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

            KNeighborsClassifier<string, int[]> knnClassifier = new KNeighborsClassifier<string, int[]>();
            List<int[]> predictedSDRsList = new List<int[]>();

            int[] activeArray = new int[numColumns];
            int maxCycles = 50;
            int currentCycle = 0;

            // 🔄 *Training Phase*
            while (!isInStableState && currentCycle < maxCycles)
            {
                Debug.WriteLine($"\n🔄 Training Cycle {currentCycle + 1}/{maxCycles} 🔄");

                foreach (var image in trainingImages)
                {
                    try
                    {
                        Debug.WriteLine($"🖼 Processing Image: {image}");

                        // ⿡ *Binarize Image*
                        string binarizedImageFile = BinarizeImageToFixedSize(image, imgSize);
                        Debug.WriteLine($"📄 Binarized Image File: {binarizedImageFile}");

                        // ⿢ *Read Input Vector*
                        int[] inputVector = ReadBinaryTextFile(binarizedImageFile);
                        Debug.WriteLine($"🔢 Input Vector Length: {inputVector.Length}");

                        // ⿣ *Compute Active Columns*
                        sp.compute(inputVector, activeArray, true);
                        var activeCols = ArrayUtils.IndexWhere(activeArray, el => el == 1);
                        Debug.WriteLine($"📊 Active Columns Count: {activeCols.Length}");

                        // ⿤ *Train KNN*
                        var activeCells = activeCols.Select(colIdx => new NeoCortexApi.Entities.Cell { Index = colIdx }).ToArray();  // Create Cell[] from active columns
                        knnClassifier.Learn(image, activeCells);  // Learning phase: learn SDRs from active columns
                        Debug.WriteLine($"🧠 KNN Learning from {image}, Stored SDR: {string.Join(",", activeCols)}");

                        // ⿥ *Store SDR*
                        predictedSDRsList.Add(activeCols);
                        string sdrFilePath = Path.Combine(sdrFolder, $"{Path.GetFileNameWithoutExtension(image)}.csv");
                        File.WriteAllText(sdrFilePath, string.Join(",", activeCols));
                        Debug.WriteLine($"💾 Stored SDR for {image} at {sdrFilePath}");
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"❌ Error processing {image}: {ex.Message}");
                    }
                }

                currentCycle++;
                Debug.WriteLine($"✅ Completed Cycle {currentCycle}.");
            }

            if (!isInStableState)
            {
                Debug.WriteLine("⚠ Training completed, but stable state not reached.");
            }
            else
            {
                Debug.WriteLine("✅ Training completed successfully.");
            }

            // *🔍 Log and Pass Predicted SDRs*
            Debug.WriteLine("\n--- PREDICTED SDRs ---");
            foreach (var sdr in predictedSDRsList)
            {
                Debug.WriteLine($"Predicted SDR: {string.Join(", ", sdr)}");
            }

            // *🔍 Log Stored SDRs Before Classification*
            Debug.WriteLine("\n--- STORED SDRs ---");
            foreach (var label in knnClassifier.StoredSDRs.Keys)
            {
                foreach (var storedSDR in knnClassifier.StoredSDRs[label])
                {
                    Debug.WriteLine($"Label: {label}, SDR: {string.Join(", ", storedSDR)}");
                }
            }

            // *🔎 Classification & Similarity Scores*
            Debug.WriteLine("\n--- CLASSIFICATION RESULTS ---");
            foreach (var sdr in predictedSDRsList)
            {
                // Convert SDR (int[]) to Cell[] before passing to GetPredictedInputValues
                var activeCells = sdr.Select(index => new NeoCortexApi.Entities.Cell { Index = index }).ToArray();

                // Get predictions (Top 5)
                var predictions = knnClassifier.GetPredictedInputValues(activeCells, 5); // Top 5 predictions

                Debug.WriteLine($"\n🔹 SDR: {string.Join(", ", sdr)}");

                foreach (var prediction in predictions)
                {
                    Debug.WriteLine($"🏷 Predicted Label: {prediction.PredictedInput} | Similarity: {prediction.Similarity:F3}");
                }
            }

            // *🔹 Pass the predicted SDRs to the restructuring function*
            RunRustructuringExperiment2(sp, predictedSDRsList);
            Debug.WriteLine("\n🔄 Running Restructuring Experiment...");

            return (sp, knnClassifier, predictedSDRsList);
        }





        /// <summary>
        /// Reconstructs images from predicted SDRs.
        /// </summary>
        private void RunRustructuringExperiment2(SpatialPooler sp, List<int[]> predictedSDRsList)
        {
            List<int[]> normalizedPermanence = new List<int[]>();

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
            }
        }
        private void RunRustructuringExperimentHtm(SpatialPooler sp, List<int[]> predictedSDRsList)
        {
            List<int[]> normalizedPermanence = new List<int[]>();
            List<double[]> similarityList = new List<double[]>();


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

                int[] inputVector = normalizePermanenceList.ToArray();


                //Calculating Similarity with encoded Inputs and Reconstructed Inputs
                var similarity = MathHelpers.JaccardSimilarityofBinaryArrays(inputVector, normalizePermanenceList.ToArray());

                double[] similarityArray = new double[] { similarity };

                //Collecting Similarity Data for visualizing
                similarityList.Add(similarityArray);
            }
            // Generate the Similarity graph using the Similarity list
            DrawSimilarityPlots(similarityList);
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

        private int[] ReadBinaryTextFile(string filePath)
        {
            var lines = File.ReadAllLines(filePath);
            return lines.SelectMany(line => line.Select(c => c == '1' ? 1 : 0)).ToArray();
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
