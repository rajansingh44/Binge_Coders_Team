using System;
using Xunit;
using ImageReconstructionwithHtm_KNNClassifier; // Reference to main project

public class ImageReconstructionSimilarityTests
{
    [Fact]
    public void Test_CosineSimilarity_IdenticalVectors()
    {
        // Arrange
        int[] vec1 = { 1, 2, 3 };
        int[] vec2 = { 1, 2, 3 };

        // Act
        double similarity = ImageReconstructionwithHtm_KNNClassifier.CosineSimilarity(vec1, vec2);

        // Assert
        Assert.Equal(1.0, similarity, 2); // Should be exactly 1 for identical vectors
    }

    [Fact]
    public void Test_CosineSimilarity_OrthogonalVectors()
    {
        // Arrange
        int[] vec1 = { 1, 0 };
        int[] vec2 = { 0, 1 };

        // Act
        double similarity = ImageReconstruction.CosineSimilarity(vec1, vec2);

        // Assert
        Assert.Equal(0.0, similarity, 2); // Orthogonal vectors should have 0 similarity
    }

    [Fact]
    public void Test_JaccardSimilarity_PartialOverlap()
    {
        // Arrange
        int[] vec1 = { 1, 2, 3, 4, 5 };
        int[] vec2 = { 3, 4, 5, 6, 7 };

        // Act
        double similarity = ImageReconstruction.JaccardSimilarity(vec1, vec2);

        // Assert
        Assert.InRange(similarity, 0, 1); // Should be a valid Jaccard similarity score
    }

    [Fact]
    public void Test_JaccardSimilarity_NoOverlap()
    {
        // Arrange
        int[] vec1 = { 1, 2, 3 };
        int[] vec2 = { 4, 5, 6 };

        // Act
        double similarity = ImageReconstruction.JaccardSimilarity(vec1, vec2);

        // Assert
        Assert.Equal(0.0, similarity, 2); // No common elements, similarity should be 0
    }
}
