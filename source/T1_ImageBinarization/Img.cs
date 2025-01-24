using NeoCortexApi;
using NeoCortexApi.Entities;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

using NeoCortexApi.Utility;
using NeoCortexApi;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NeoCortex;

namespace T1_ImageBinarization
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


            // The Size of the Image Height and width is 32 pixel

            int imageSize = 32;
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
            var sp = RunExperiment(cfg, inputPrefix);
            //Runing the Reconstruction Method Experiment
            RunRustructuringExperiment(sp);

        }

        /// <summary>
        /// Implements the experiment.
        /// </summary>
        /// <param name="cfg"></param>
        /// <param name="inputPrefix"> The name of the images</param>
        /// <returns>The trained bersion of the SP.</returns>
        private SpatialPooler RunExperiment(HtmConfig cfg, string inputPrefix)
        {

            var mem = new Connections(cfg);

            bool isInStableState = false;

            int numColumns = 64 * 64;
            //Accessing the Image Folder form the Cureent Directory
            string trainingFolder = "Sample\\TestFiles";
            //Accessing the Image Folder form the Cureent Directory Foldfer
            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");
            //Image Size
            int imageSize = 32;
            //Folder Name in the Directorty 
            string testName = "test_image";

            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 50, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                // Event should only be fired when entering the stable state.
                if (isStable)
                {
                    isInStableState = true;
                    Debug.WriteLine($"Entered STABLE state: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
                }
                else
                {
                    isInStableState = false;
                    Debug.WriteLine($"INSTABLE STATE");
                }
                // Ideal SP should never enter unstable state after stable state.
                Debug.WriteLine($"Entered STABLE state: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
            }, requiredSimilarityThreshold: 0.975);

            // It creates the instance of Spatial Pooler Multithreaded version.
            SpatialPooler sp = new SpatialPooler(hpa);

            //Initializing the Spatial Pooler Algorithm
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

            //Image Size
            int imgSize = 32;
            int[] activeArray = new int[numColumns];

            int numStableCycles = 0;
            // Runnig the Traning Cycle for 5 times
            int maxCycles = 5;
            int currentCycle = 0;

            while (!isInStableState && currentCycle < maxCycles)
            {
                foreach (var Image in trainingImages)
                {
                    //Binarizing the Images before taking Inputs for the Sp
                    string inputBinaryImageFile = NeoCortexUtils.BinarizeImage($"{Image}", imgSize, testName);

                    // Read Binarized and Encoded input csv file into array
                    int[] inputVector = NeoCortexUtils.ReadCsvIntegers(inputBinaryImageFile).ToArray();

                    int[] oldArray = new int[activeArray.Length];
                    List<double[,]> overlapArrays = new List<double[,]>();
                    List<double[,]> bostArrays = new List<double[,]>();

                    sp.compute(inputVector, activeArray, true);
                    //Getting the Active Columns
                    var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);

                    Debug.WriteLine($"'Cycle: {currentCycle} - Image-Input: {Image}'");
                    Debug.WriteLine($"INPUT :{Helpers.StringifyVector(inputVector)}");
                    Debug.WriteLine($"SDR:{Helpers.StringifyVector(activeCols)}");
                    Debug.WriteLine($"OUTPUT:{(inputBinaryImageFile)}\n");
                }

                currentCycle++;

                // Check if the desired number of cycles is reached
                if (currentCycle >= maxCycles)
                    break;

                // Increment numStableCycles only when it's in a stable state
                if (isInStableState)
                    numStableCycles++;
            }

            return sp;
        }
        /// <summary>
        /// Runs the restructuring experiment using the provided spatial pooler. 
        /// This method iterates through a set of training images, computes spatial pooling, 
        /// reconstructs permanence values, and generates heatmaps and similarity graphs based on the results.
        /// </summary>
        /// <param name="sp">The spatial pooler to use for the experiment.</param>
        private void RunRustructuringExperiment(SpatialPooler sp)
        {
            // Path to the folder containing training images
            string trainingFolder = "Sample\\TestFiles";

            // Path to the archive folder for storing test image files
            string archiveFolder = Path.Combine(trainingFolder, "archive");

            // Ensure the archive folder exists
            Directory.CreateDirectory(archiveFolder);

            // //Path for the output of Binarized Image
            string output = Path.GetFullPath("output");

            Directory.CreateDirectory(output);

            // Get all image files matching the specified prefix
            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");

            // Size of the images
            int imgSize = 32;

            // Array to hold active columns
            int[] activeArray = new int[64 * 64];

            // Lists to store data for visualization
            List<List<double>> heatmapData = new List<List<double>>();
            List<int[]> binarizedEncodedInputs = new List<int[]>();
            List<int[]> normalizedPermanence = new List<int[]>();
            List<double[]> similarityList = new List<double[]>();

            // Iterate through each training image
            foreach (var image in trainingImages)
            {
                // Generate a unique test image name for each input
                string testImageName = Path.GetFileNameWithoutExtension(image) + "_test_image";

                // Create a binarized version of the input image and store the output file
                string inputBinaryImageFile = NeoCortexUtils.BinarizeImage(image, imgSize, testImageName);
                // File.Create(inputBinaryImageFile).Close();

                //File.WriteAllText(output, inputBinaryImageFile);
                // Read input CSV file into an array
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

                // Add all reconstructed permanence values to the dictionary
                foreach (var kvp in reconstructedPermanence)
                {
                    int inputIndex = kvp.Key;
                    double probability = kvp.Value;

                    // Use the existing probability
                    allPermanenceDictionary[inputIndex] = probability;
                }

                // Assign inactive columns a permanence value of 0
                for (int inputIndex = 0; inputIndex < maxInput; inputIndex++)
                {
                    if (!reconstructedPermanence.ContainsKey(inputIndex))
                    {
                        allPermanenceDictionary[inputIndex] = 0.0;
                    }
                }

                // Sort the dictionary by keys
                var sortedAllPermanenceDictionary = allPermanenceDictionary.OrderBy(kvp => kvp.Key);

                // Convert the sorted dictionary of permanence values to a list
                List<double> permanenceValuesList = sortedAllPermanenceDictionary.Select(kvp => kvp.Value).ToList();

                // Collect heatmap data for visualization
                heatmapData.Add(permanenceValuesList);

                // Collect encoded data for visualization
                binarizedEncodedInputs.Add(inputVector);

                // Normalize permanence values based on a threshold
                var thresholdValue = 30.5;
                List<int> normalizePermanenceList = Helpers.ThresholdingProbabilities(permanenceValuesList, thresholdValue);

                // Collect normalized permanence data for visualization
                normalizedPermanence.Add(normalizePermanenceList.ToArray());

                // Calculate similarity between the encoded inputs and reconstructed inputs
                var similarity = MathHelpers.JaccardSimilarityofBinaryArrays(inputVector, normalizePermanenceList.ToArray());
                double[] similarityArray = new double[] { similarity };

                // Collect similarity data for visualization
                similarityList.Add(similarityArray);

                // Save the test image data to a .txt file in the archive folder
                string testImageFilePathp = Path.Combine(archiveFolder, $"{testImageName}.txt");
                File.WriteAllLines(testImageFilePathp, inputVector.Select(val => val.ToString()));


                //output

                //inputBinaryImageFile = Path.Combine(output, $"{testImageName}.txt");

            }

            // Generate the 1D heatmaps using the heatmap data list
            Generate1DHeatmaps(heatmapData, binarizedEncodedInputs, normalizedPermanence);

            // Generate the similarity graph using the similarity list
            DrawSimilarityPlots(similarityList);
        }


        /// <summary>
        /// Generates 1D heatmaps based on the provided heatmap data and normalized permanence values.
        /// </summary>
        /// <param name="heatmapData">List of lists containing heatmap data.</param>
        /// <param name="normalizedPermanence">List of arrays containing normalized permanence values.</param>
        private void Generate1DHeatmaps(List<List<double>> heatmapData, List<int[]> normalizedPermanence, List<int[]> BinarizedencodedInputs)
        {
            int i = 1;

            foreach (var values in heatmapData)
            {
                // Define the folder path based on your requirements
                string folderPath = Path.Combine(Environment.CurrentDirectory, "1DHeatMap_Image_Inputs");

                // Create the folder if it doesn't exist
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }
                // Define the file path with the folder path
                string filePath = Path.Combine(folderPath, $"heatmap_{i}.png");
                Debug.WriteLine($"FilePath: {filePath}");

                // Convert the probabilitiesList to a 1D array using ToArray
                double[] array1D = values.ToArray();

                // Call the  Draw1DHeatmap function with the dynamically generated file path along with all necessary Perameters
                NeoCortexUtils.Draw1dHeatmap(new List<double[]>() { array1D }, new List<int[]>() { normalizedPermanence[i - 1] }, new List<int[]>() { BinarizedencodedInputs[i - 1] }, filePath, 1024, 15, 30, 15, 5, 30);

                Debug.WriteLine("Heatmap generated and saved successfully.");
                i++;
            }

        }

        // <summary>
        /// Draws a combined similarity plot based on the provided list of arrays containing similarity values.
        /// The combined similarity plot is generated by combining all similarity values from the list of arrays,
        /// creating a single list of similarities, and then drawing the plot.
        /// </summary>
        /// <param name="similaritiesList">List of arrays containing similarity values.</param>
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
            //     NeoCortexUtils.DrawCombinedSimilarityPlot(combinedSimilarities, filePath, 1000, 850);

            //      Debug.WriteLine($"Combined similarity plot generated and saved successfully.");

        }
    }
}
