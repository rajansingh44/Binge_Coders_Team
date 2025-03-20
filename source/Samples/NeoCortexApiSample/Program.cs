using ExcelDataReader;
using NeoCortexApi;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using Newtonsoft.Json.Linq;
using OfficeOpenXml.FormulaParsing.LexicalAnalysis;
using Org.BouncyCastle.Ocsp;
using ScottPlot.Drawing.Colormaps;
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

         /// <summary>
        ///This class implements an **image reconstruction pipeline** 
        ///using **HTM (Hierarchical Temporal Memory) and KNN (K-Nearest Neighbors)** classifiers. 
        ///It processes images, extracts Spatial Pooler(SP) representations, trains classifiers, and reconstructs images based on predictions.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {

            ImageReconstructionwithHtm_KNNClassifier exp = new ImageReconstructionwithHtm_KNNClassifier();
            exp.Run();
        }
    }
}

