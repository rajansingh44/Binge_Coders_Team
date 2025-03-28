using System;
using System.Globalization;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeoCortexApi;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Network;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;


namespace MultiSequenceLearning
{
    /// <summary>
    /// Implements an experiment that demonstrates how to learn sequences.
    /// </summary>
    public class MultiSequenceLearning
    {
        /// <summary>
        /// Runs the learning of sequences.
        /// </summary>
        /// <param name="sequences">Dictionary of sequences. KEY is the sewuence name, the VALUE is th elist of element of the sequence.</param>
        public Predictor Run(List<Sequence> sequences)
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(MultiSequenceLearning)}");

            int inputBits = 100;
            int numColumns = 1024;

            HtmConfig cfg = HelperMethods.FetchHTMConfig(inputBits, numColumns);

            EncoderBase encoder = HelperMethods.GetEncoder(inputBits);

            return RunExperiment(inputBits, cfg, encoder, sequences);
        }

        /// <summary>
        /// Runs an experiment using the HTM (Hierarchical Temporal Memory) model, training a Spatial Pooler (SP) 
        /// and Temporal Memory (TM) on given input sequences to classify and predict patterns.
        /// </summary>
        /// <param name="inputBits">The number of input bits for the encoder.</param>
        /// <param name="cfg">The HTM configuration settings.</param>
        /// <param name="encoder">The encoder used for processing input sequences.</param>
        /// <param name="sequences">A list of input sequences for training and testing.</param>
        /// <returns>
        /// A <see cref="Predictor"/> object containing the trained HTM model, connections, and classifier.
        /// </returns>
        private Predictor RunExperiment(int inputBits, HtmConfig cfg, EncoderBase encoder, List<Sequence> sequences)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            int maxMatchCnt = 0;
            var mem = new Connections(cfg);
            bool isInStableState = false;

            HtmClassifier<string, ComputeCycle> cls = new HtmClassifier<string, ComputeCycle>();

            // Determine the number of unique inputs in the given sequences.
            var numUniqueInputs = GetNumberOfInputs(sequences);

            CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");
            TemporalMemory tm = new TemporalMemory();

            Console.WriteLine("------------ START ------------");

            // Initialize Homeostatic Plasticity Controller (HPC) to track stability of the Spatial Pooler (SP).
            HomeostaticPlasticityController hpc = new HomeostaticPlasticityController(mem, numUniqueInputs * 150, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                if (isStable)
                    Debug.WriteLine($"STABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
                else
                    Debug.WriteLine($"INSTABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");

                isInStableState = isStable;
            }, numOfCyclesToWaitOnChange: 50);

            SpatialPoolerMT sp = new SpatialPoolerMT(hpc);
            sp.Init(mem);
            tm.Init(mem);

            // Adding encoder and spatial pooler to the HTM layer
            layer1.HtmModules.Add("encoder", encoder);
            layer1.HtmModules.Add("sp", sp);

            int cycle = 0;
            int matches = 0;
            var lastPredictedValues = new List<string>(new string[] { "0" });
            int maxCycles = 3500;

            // Train Spatial Pooler until it reaches a stable state.
            for (int i = 0; i < maxCycles && !isInStableState; i++)
            {
                matches = 0;
                cycle++;

                Debug.WriteLine($"-------------- Newborn SP Cycle {cycle} ---------------");
                Console.WriteLine($"-------------- Newborn SP Cycle {cycle} ---------------");

                foreach (var inputs in sequences)
                {
                    foreach (var input in inputs.data)
                    {
                        Debug.WriteLine($" -- {inputs.name} - {input} --");

                        var lyrOut = layer1.Compute(input, true);

                        if (isInStableState)
                            break;
                    }

                    if (isInStableState)
                        break;
                }
            }

            // Reset classifier state before training with TM.
            cls.ClearState();

            // Activate Temporal Memory algorithm and add it to the HTM layer.
            layer1.HtmModules.Add("tm", tm);

            // Train SP+TM with sequences.
            foreach (var sequenceKeyPair in sequences)
            {
                Debug.WriteLine($"-------------- Sequences {sequenceKeyPair.name} ---------------");
                Console.WriteLine($"-------------- Sequences {sequenceKeyPair.name} ---------------");

                int maxPrevInputs = sequenceKeyPair.data.Length - 1;
                List<string> previousInputs = new List<string> { "-1" };

                for (int i = 0; i < maxCycles; i++)
                {
                    matches = 0;
                    cycle++;

                    Debug.WriteLine($"-------------- Cycle SP+TM {cycle} ---------------");
                    Console.WriteLine($"-------------- Cycle SP+TM {cycle} ---------------");

                    foreach (var input in sequenceKeyPair.data)
                    {
                        Debug.WriteLine($"-------------- {input} ---------------");

                        var lyrOut = layer1.Compute(input, true) as ComputeCycle;
                        var activeColumns = layer1.GetResult("sp") as int[];

                        previousInputs.Add(input.ToString());
                        if (previousInputs.Count > (maxPrevInputs + 1))
                            previousInputs.RemoveAt(0);

                        if (previousInputs.Count < maxPrevInputs)
                            continue;

                        string key = GetKey(previousInputs, input, sequenceKeyPair.name);
                        List<Cell> actCells = (lyrOut.ActiveCells.Count == lyrOut.WinnerCells.Count) ? lyrOut.ActiveCells : lyrOut.WinnerCells;

                        cls.Learn(key, actCells.ToArray());

                        Debug.WriteLine($"Col SDR: {Helpers.StringifyVector(lyrOut.ActivColumnIndicies)}");
                        Debug.WriteLine($"Cell SDR: {Helpers.StringifyVector(actCells.Select(c => c.Index).ToArray())}");

                        if (lastPredictedValues.Contains(key))
                        {
                            matches++;
                            Debug.WriteLine($"Match. Actual value: {key} - Predicted value: {lastPredictedValues.FirstOrDefault(key)}.");
                        }
                        else
                        {
                            Debug.WriteLine($"Mismatch! Actual value: {key} - Predicted values: {String.Join(',', lastPredictedValues)}");
                        }

                        if (lyrOut.PredictiveCells.Count > 0)
                        {
                            var predictedInputValues = cls.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 3);
                            foreach (var item in predictedInputValues)
                            {
                                Debug.WriteLine($"Current Input: {input} \t| Predicted Input: {item.PredictedInput} - {item.Similarity}");
                            }

                            lastPredictedValues = predictedInputValues.Select(v => v.PredictedInput).ToList();
                        }
                        else
                        {
                            Debug.WriteLine($"NO CELLS PREDICTED for next cycle.");
                            lastPredictedValues = new List<string>();
                        }
                    }

                    double maxPossibleAccuracy = (double)(sequenceKeyPair.data.Length - 1) / sequenceKeyPair.data.Length * 100.0;
                    double accuracy = (double)matches / sequenceKeyPair.data.Length * 100.0;

                    Debug.WriteLine($"Cycle: {cycle}\tMatches={matches} of {sequenceKeyPair.data.Length}\t {accuracy}%");
                    Console.WriteLine($"Cycle: {cycle}\tMatches={matches} of {sequenceKeyPair.data.Length}\t {accuracy}%");

                    if (accuracy >= maxPossibleAccuracy)
                    {
                        maxMatchCnt++;
                        Debug.WriteLine($"100% accuracy reached {maxMatchCnt} times.");

                        if (maxMatchCnt >= 30)
                        {
                            sw.Stop();
                            Debug.WriteLine($"Sequence learned. Stable state reached after 30 repeats with accuracy {accuracy}. Elapsed time: {sw.Elapsed}.");
                            break;
                        }
                    }
                    else if (maxMatchCnt > 0)
                    {
                        Debug.WriteLine($"Accuracy dropped after {maxMatchCnt} repeats at 100% accuracy. Learning will continue.");
                        maxMatchCnt = 0;
                    }

                    // Reset TM state to ensure learning consistency.
                    tm.Reset(mem);
                }
            }

            Debug.WriteLine("------------ END ------------");

            return new Predictor(layer1, mem, cls);
        }



        /// <summary>
        /// Gets the number of all unique inputs.
        /// </summary>
        /// <param name="sequences">Alle sequences.</param>
        /// <returns></returns>
        private int GetNumberOfInputs(List<Sequence> sequences)
        {
            int num = 0;

            foreach (var inputs in sequences)
            {
                //num += inputs.Value.Distinct().Count();
                num += inputs.data.Length;
            }

            return num;
        }


        /// <summary>
        /// Constracts the unique key of the element of an sequece. This key is used as input for HtmClassifier.
        /// It makes sure that alle elements that belong to the same sequence are prefixed with the sequence.
        /// The prediction code can then extract the sequence prefix to the predicted element.
        /// </summary>
        /// <param name="prevInputs"></param>
        /// <param name="input"></param>
        /// <param name="sequence"></param>
        /// <returns></returns>
        private static string GetKey(List<string> prevInputs, double input, string sequence)
        {
            string key = String.Empty;

            for (int i = 0; i < prevInputs.Count; i++)
            {
                if (i > 0)
                    key += "-";

                key += (prevInputs[i]);
            }
            //Console.WriteLine($"GetKey={sequence}_{key}");
            return $"{sequence}_{key}";
        }
    }
}
