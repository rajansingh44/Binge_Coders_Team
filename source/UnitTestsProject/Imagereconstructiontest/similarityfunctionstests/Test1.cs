using Microsoft.VisualStudio.TestTools.UnitTesting;
using similarityFunctions;

namespace similarityfunctionstests
{
    [TestClass]
    public sealed class Test1
    {
        [TestMethod]
        public void TestCosineSimilarity_ExactlysameVectors()
        {
            int[] vec1 = { 1, 0, 1, 0, 1 };
            int[] vec2 = { 1, 0, 1, 0, 1 };
            double result = calculatesimilarity.CosineSimilarity(vec1, vec2);
            Assert.AreEqual(1.0, result, 0.01);
        }

        [TestMethod]
        public void TestCosineSimilarity_CompletelyDifferentVectors()
        {
            int[] vec1 = { 1, 1, 1, 1, 1 };
            int[] vec2 = { 0, 0, 0, 0, 0 };
            double result = calculatesimilarity.CosineSimilarity(vec1, vec2);
            Assert.AreEqual(0.0, result, 0.01);
        }

        [TestMethod]
        public void TestCosineSimilarity_PartiallySimilarVectors()
        {
            int[] vec1 = { 1, 0, 1, 1, 0 };
            int[] vec2 = { 1, 1, 1, 0, 0 };
            double result = calculatesimilarity.CosineSimilarity(vec1, vec2);
            Assert.AreEqual(0.67, result, 0.01);
        }


        [TestMethod]
        public void TestConvertPermanenceToBinary()
        {
            // Arrange
            double[] permanenceValues = { 0.5, 0.8, 0.3, 0.9, 0.7 };
            double threshold = 0.7;
            int[] expectedBinaryArray = { 0, 1, 0, 1, 1 };

            // Act
            int[] result = calculatesimilarity.ConvertPermanenceToBinary(permanenceValues, threshold);


            // Assert
            CollectionAssert.AreEqual(expectedBinaryArray, result);
        }

        [TestMethod]
        public void Test_AddActiveColsToStoredSDRs()
        {
            // Arrange
            var image = "image1";  // The image identifier
            var activeCols = new int[] { 1, 0, 1, 0 };  // The active columns array to be added
            var storedSDRs = new Dictionary<string, List<int[]>>();  // The dictionary that will store the SDRs

            // Act
            // Call the method to add activeCols to the storedSDRs for the image
            calculatesimilarity.AddActiveColsToStoredSDRs(image, activeCols, storedSDRs);

            // Assert
            // Check if the dictionary contains the image as a key
            Assert.IsTrue(storedSDRs.ContainsKey(image), "StoredSDRs should contain the image key.");

            // Check if the list for the image contains exactly one entry
            Assert.AreEqual(1, storedSDRs[image].Count, "StoredSDRs should contain one entry for the image.");

            // Check if the first entry in the list is equal to the activeCols array
            CollectionAssert.AreEqual(activeCols, storedSDRs[image][0], "The activeCols array should be correctly added to the list.");
        }

        [TestMethod]
        public void TestConvertPermanenceToBinary_NoNegativeValues_ReturnsBinaryArray()
        {
            // Arrange
            double[] permanenceValues = { 0.5, 0.8, 0.3, 0.9, 0.7 };  // All non-negative values
            double threshold = 0.7;
            int[] expectedBinaryArray = { 0, 1, 0, 1, 1 };

            // Act
            int[] result = calculatesimilarity.ConvertPermanenceToBinary_negative(permanenceValues, threshold);

            // Assert
            CollectionAssert.AreEqual(expectedBinaryArray, result);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException), "Permanence values cannot be negative.")]
        public void TestConvertPermanenceToBinary_WithNegativeValue_ThrowsException()
        {
            // Arrange
            double[] permanenceValues = { 0.5, 0.8, -0.3, 0.9, 0.7 };  // Includes a negative value
            double threshold = 0.7;

            // Act
            // This should throw an ArgumentException due to the negative value
            int[] result = calculatesimilarity.ConvertPermanenceToBinary_negative(permanenceValues, threshold);

            // Assert: The exception is expected, so no further assertions are necessary.
        }
    }
}
