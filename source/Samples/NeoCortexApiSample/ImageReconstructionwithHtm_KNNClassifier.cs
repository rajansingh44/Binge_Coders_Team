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

            while (!isInStableState && currentCycle < maxCycles)
            {
                Debug.WriteLine($"\n🔄 Training Cycle {currentCycle + 1}/{maxCycles} 🔄");

                foreach (var image in trainingImages)
                {
                    try
                    {
                        Debug.WriteLine($"🖼 Processing Image: {image}");

                        // Binarize Image
                        string binarizedImageFile = BinarizeImageToFixedSize(image, imgSize);
                        Debug.WriteLine($"📄 Binarized Image File: {binarizedImageFile}");

                        // Read Input Vector
                        int[] inputVector = ReadBinaryTextFile(binarizedImageFile);
                        Debug.WriteLine($"🔢 Input Vector Length: {inputVector.Length}");

                        // Compute Active Columns
                        sp.compute(inputVector, activeArray, true);
                        var activeCols = ArrayUtils.IndexWhere(activeArray, el => el == 1);
                        Debug.WriteLine($"📊 Active Columns Count: {activeCols.Length}");

                        // Train KNN
                        var activeCells = activeCols.Select(colIdx => new NeoCortexApi.Entities.Cell { Index = colIdx }).ToArray();
                        knnClassifier.Learn(image, activeCells);
                        Debug.WriteLine($"🧠 KNN Learning from {image}, Stored SDR: {string.Join(",", activeCols)}");

                        // Store SDR
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

            return (sp, knnClassifier, predictedSDRsList);
        }

        private void RunRustructuringExperiment2(SpatialPooler sp, List<int[]> predictedSDRsList)
        {
            List<int[]> normalizedPermanence = new List<int[]>();

            foreach (var predictedSDR in predictedSDRsList)
            {
                Debug.WriteLine("Reconstructing permanence for SDR...");

                Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(predictedSDR);

                Dictionary<int, double> allPermanenceDictionary = new Dictionary<int, double>();
                foreach (var kvp in reconstructedPermanence)
                {
                    allPermanenceDictionary[kvp.Key] = kvp.Value;
                }

                int imgsize = 52 * 52;

                for (int inputIndex = 0; inputIndex < imgsize; inputIndex++)
                {
                    if (!reconstructedPermanence.ContainsKey(inputIndex))
                    {
                        allPermanenceDictionary[inputIndex] = 0.0;
                    }
                }

                var ThresholdValue = 67.0;
                List<double> permanenceValuesList = allPermanenceDictionary.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();
                List<int> normalizePermanenceList = Helpers.ThresholdingforResetImg(permanenceValuesList, ThresholdValue);

                normalizedPermanence.Add(normalizePermanenceList.ToArray());

                string outputPath = $"ReconstructedSDR_{predictedSDRsList.IndexOf(predictedSDR)}";
                NeoCortexUtils.SaveBinarizedImageFromBinaryArray(normalizePermanenceList.ToArray(), outputPath);
                Debug.WriteLine($"Reconstructed Image saved at {outputPath}");
            }
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

                File.WriteAllLines(outputFile, binaryArray.Select((b, i) => (i % gridSize == 0 ? "\n" : "") + b));
            }

            return outputFile;
        }
    }
}
