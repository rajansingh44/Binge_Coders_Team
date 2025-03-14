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
    }
}
