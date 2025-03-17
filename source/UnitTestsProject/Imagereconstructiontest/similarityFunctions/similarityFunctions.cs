using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace similarityFunctions
{
    public class calculatesimilarity
    {
        public static double CosineSimilarity(int[] vec1, int[] vec2)
        {
            if (vec1 == null || vec2 == null || vec1.Length != vec2.Length)
                throw new ArgumentException("Vectors must be non-null and of the same length.");

            double dotProduct = 0, magnitude1 = 0, magnitude2 = 0;

            for (int i = 0; i < vec1.Length; i++)
            {
                dotProduct += vec1[i] * vec2[i];
                magnitude1 += vec1[i] * vec1[i];
                magnitude2 += vec2[i] * vec2[i];
            }

            return magnitude1 == 0 || magnitude2 == 0 ? 0 : dotProduct / (Math.Sqrt(magnitude1) * Math.Sqrt(magnitude2));
        }

        public static int[] ConvertPermanenceToBinary(double[] permanenceValues, double threshold)
        {
            if (permanenceValues == null || permanenceValues.Length == 0)
                throw new ArgumentException("Permanence values cannot be null or empty.");

            int[] binaryArray = new int[permanenceValues.Length];

            for (int i = 0; i < permanenceValues.Length; i++)
            {
                binaryArray[i] = permanenceValues[i] >= threshold ? 1 : 0;
            }

            return binaryArray;
        }

        public static void AddActiveColsToStoredSDRs(string image, int[] activeCols, Dictionary<string, List<int[]>> storedSDRs)
        {
            if (!storedSDRs.ContainsKey(image))
            {
                storedSDRs[image] = new List<int[]>();
            }
            storedSDRs[image].Add(activeCols);
        }

        public static int[] ConvertPermanenceToBinary_negative(double[] permanenceValues, double threshold)
        {
            if (permanenceValues == null || permanenceValues.Length == 0)
                throw new ArgumentException("Permanence values cannot be null or empty.");

            // Check for negative permanence values
            foreach (var value in permanenceValues)
            {
                if (value < 0)
                {
                    throw new ArgumentException("Permanence values cannot be negative.");
                }
            }

            int[] binaryArray = new int[permanenceValues.Length];

            for (int i = 0; i < permanenceValues.Length; i++)
            {
                binaryArray[i] = permanenceValues[i] >= threshold ? 1 : 0;
            }

            return binaryArray;
        }


    }
}
