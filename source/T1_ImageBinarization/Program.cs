//sprint 1 changes
using Newtonsoft.Json.Linq;
using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Xml;


namespace Imagereconstruction
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {

                // Providing multiple images in a single folder as an input 
                string InputforBinarizer = Path.Combine(Directory.GetCurrentDirectory(), "Inputs");

                // This program is capable of Multiple Inputs
                int[] binarizedImage = ImgBinarize.ConvertImagesToBinaryForm(InputforBinarizer);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Following error is: {ex.Message}");
            }
        }
    }
}
