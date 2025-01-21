using ExcelDataReader;
using NeoCortexApi;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using Newtonsoft.Json.Linq;
using OfficeOpenXml.FormulaParsing.LexicalAnalysis;
using Org.BouncyCastle.Ocsp;
using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Xml;
using static NeoCortexApiSample.MultisequenceLearningTeamMSL;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace NeoCortexApiSample
{
    class Program
    {
        static double MinVal = 0.0;
        static double MaxVal = 99.0;

        /// <summary>
        /// This sample shows a typical experiment code for SP and TM.
        /// You must start this code in debugger to follow the trace.
        /// and TM.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            //
            // Starts experiment that demonstrates how to learn spatial patterns.
            //SpatialPatternLearning experiment = new SpatialPatternLearning();
            //experiment.Run();

            // Starts experiment that demonstrates how to learn spatial patterns
            //ImageBinarizerSpatialPattern exp = new ImageBinarizerSpatialPattern();
            //exp.Run();


            ImageReconstructionwithHtm_KNNClassifier exp = new ImageReconstructionwithHtm_KNNClassifier();
            exp.Run();
            //
            // Starts experiment that demonstrates how to learn spatial patterns.
            // SequenceLearning experiment = new SequenceLearning();
            //experiment.Run();

            // This method is developed by Team_MSL to read arbitrary data from single txt file and improve CPU utilization*/
            //   RunPredictionMultiSequenceExperiment();
        }