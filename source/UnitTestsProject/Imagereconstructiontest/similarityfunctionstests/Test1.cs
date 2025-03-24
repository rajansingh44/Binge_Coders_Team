using Microsoft.VisualStudio.TestTools.UnitTesting;
using similarityFunctions;
using System;
using System.Collections.Generic;

namespace similarityfunctionstests
{
    /// <summary>
    /// Unit tests for the <see cref="calculatesimilarity"/> class.
    /// This class tests various similarity functions and permanence-to-binary conversions.
    /// </summary>
    [TestClass]
    public sealed class Test1
    {
        /// <summary>
        /// Tests the <see cref="calculatesimilarity.CosineSimilarity"/> method
        /// by passing two exactly identical vectors.
        /// </summary>
        [TestMethod]
        public void TestCosineSimilarity_ExactlysameVectors()
        {
            // Arrange
            int[] vec1 = { 1, 0, 1, 0, 1 };
            int[] vec2 = { 1, 0, 1, 0, 1 };

            // Act
            double result = calculatesimilarity.CosineSimilarity(vec1, vec2);

            // Assert
            Assert.AreEqual(1.0, result, 0.01, "Cosine similarity should be 1.0 for identical vectors.");
        }

        /// <summary>
        /// Tests the <see cref="calculatesimilarity.CosineSimilarity"/> method
        /// with two completely different vectors.
        /// </summary>
        [TestMethod]
        public void TestCosineSimilarity_CompletelyDifferentVectors()
        {
            // Arrange
            int[] vec1 = { 1, 1, 1, 1, 1 };
            int[] vec2 = { 0, 0, 0, 0, 0 };

            // Act
            double result = calculatesimilarity.CosineSimilarity(vec1, vec2);

            // Assert
            Assert.AreEqual(0.0, result, 0.01, "Cosine similarity should be 0.0 for orthogonal vectors.");
        }

        /// <summary>
        /// Tests the <see cref="calculatesimilarity.CosineSimilarity"/> method
        /// with partially similar vectors.
        /// </summary>
        [TestMethod]
        public void TestCosineSimilarity_PartiallySimilarVectors()
        {
            // Arrange
            int[] vec1 = { 1, 0, 1, 1, 0 };
            int[] vec2 = { 1, 1, 1, 0, 0 };

            // Act
            double result = calculatesimilarity.CosineSimilarity(vec1, vec2);

            // Assert
            Assert.AreEqual(0.67, result, 0.01, "Expected cosine similarity is approximately 0.67.");
        }

        /// <summary>
        /// Tests the <see cref="calculatesimilarity.ConvertPermanenceToBinary"/> method
        /// to check if permanence values are correctly converted to binary.
        /// </summary>
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
            CollectionAssert.AreEqual(expectedBinaryArray, result, "Permanence values should be correctly converted to binary.");
        }

        /// <summary>
        /// Tests the <see cref="calculatesimilarity.AddActiveColsToStoredSDRs"/> method
        /// by adding an active column SDR to the stored SDR dictionary.
        /// </summary>
        [TestMethod]
        public void Test_AddActiveColsToStoredSDRs()
        {
            // Arrange
            var image = "image1";  // The image identifier
            var activeCols = new int[] { 1, 0, 1, 0 };  // The active columns array to be added
            var storedSDRs = new Dictionary<string, List<int[]>>();  // The dictionary that will store the SDRs

            // Act
            calculatesimilarity.AddActiveColsToStoredSDRs(image, activeCols, storedSDRs);

            // Assert
            Assert.IsTrue(storedSDRs.ContainsKey(image), "StoredSDRs should contain the image key.");
            Assert.AreEqual(1, storedSDRs[image].Count, "StoredSDRs should contain one entry for the image.");
            CollectionAssert.AreEqual(activeCols, storedSDRs[image][0], "The activeCols array should be correctly added to the list.");
        }

        /// <summary>
        /// Tests the <see cref="calculatesimilarity.ConvertPermanenceToBinary_negative"/> method
        /// to check conversion when no negative values are present.
        /// </summary>
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
            CollectionAssert.AreEqual(expectedBinaryArray, result, "Permanence values should be converted to binary correctly.");
        }

        /// <summary>
        /// Tests the <see cref="calculatesimilarity.ConvertPermanenceToBinary_negative"/> method
        /// to verify that an exception is thrown when negative permanence values are provided.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(ArgumentException), "Permanence values cannot be negative.")]
        public void TestConvertPermanenceToBinary_WithNegativeValue_ThrowsException()
        {
            // Arrange
            double[] permanenceValues = { 0.5, 0.8, -0.3, 0.9, 0.7 };  // Includes a negative value
            double threshold = 0.7;

            // Act
            calculatesimilarity.ConvertPermanenceToBinary_negative(permanenceValues, threshold);

            // Assert: The exception is expected, so no further assertions are necessary.
        }
    }
}
