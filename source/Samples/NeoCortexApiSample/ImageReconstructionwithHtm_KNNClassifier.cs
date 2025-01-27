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
            bool isInStableState = false;  //Mausam


            int numColumns = 64 * 64;
            string trainingFolder = "Sample\\TestFiles";
            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");
            int imgSize = 28;
            string testName = "test_image";

            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 50, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                isInStableState = isStable;
                Debug.WriteLine(isStable ? "Entered STABLE state." : "INSTABLE STATE.");
            }, requiredSimilarityThreshold: 0.975); // Pradeep 26/01 

            SpatialPooler sp = new SpatialPooler(hpa);
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) }); //Rajan



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