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

        public void Run()
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(ImageReconstructionwithHtm_KNNClassifier)}");

            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            int numColumns = 64 * 64;
            int imageSize = 28;
            var colDims = new int[] { 64, 64 };

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

            var sp = RunExperimentWithKNNClassifier(cfg, inputPrefix);
        }

        private (SpatialPooler, KNeighborsClassifier<string, int[]>) RunExperimentWithKNNClassifier(HtmConfig cfg, string inputPrefix)
        {
            var mem = new Connections(cfg);
            bool isInStableState = false;

            int numColumns = 64 * 64;
            string trainingFolder = "Sample\\TestFiles";
            string outputFolder = "Output";
            Directory.CreateDirectory(outputFolder);

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

            while (!isInStableState && currentCycle < maxCycles)
            {
                foreach (var image in trainingImages)
                {
                    string inputBinaryImageFile = NeoCortexUtils.BinarizeImage($"{image}", imgSize, testName);
                    int[] inputVector = NeoCortexUtils.ReadCsvIntegers(inputBinaryImageFile).ToArray();

                    sp.compute(inputVector, activeArray, true);
                    var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                    var activeCells = activeCols.Select(colIdx => new Cell { Index = colIdx }).ToArray();
                    knnClassifier.Learn(image, activeCells);

                    Debug.WriteLine($"Cycle: {currentCycle} - Image-Input: {image}");
                }

                currentCycle++;
                if (currentCycle >= maxCycles)
                    break;
            }

            string testImage = trainingImages[0];
            string testBinaryImageFile = NeoCortexUtils.BinarizeImage($"{testImage}", imgSize, testName);
            int[] testInputVector = NeoCortexUtils.ReadCsvIntegers(testBinaryImageFile).ToArray();

            sp.compute(testInputVector, activeArray, false);
            var testActiveCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);
            var testActiveCells = testActiveCols.Select(colIdx => new Cell { Index = colIdx }).ToArray();

            var predictions = knnClassifier.GetPredictedInputValues(testActiveCells, 7);
            foreach (var prediction in predictions)
            {
                Debug.WriteLine($"Predicted Label: {prediction.PredictedInput}, Accuracy: {prediction.Similarity}");
            }

            return (sp, knnClassifier);
        }
    }
}
