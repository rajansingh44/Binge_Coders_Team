using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using NeoCortexApi.Entities;

namespace NeoCortexApi.Classifiers
{
    public static class EnumExtension
    {
        public static IEnumerable<(T item, int index)> WithIndex<T>(this IEnumerable<T> self)
            => self.Select((item, index) => (item, index));
    }

    public class DefaultDictionary<TKey, TValue> : Dictionary<TKey, TValue> where TValue : new()
    {
        public new TValue this[TKey key]
        {
            get
            {
                if (!TryGetValue(key, out TValue val))
                {
                    val = new TValue();
                    Add(key, val);
                }
                return val;
            }
            set => base[key] = value;
        }
    }

    public class ClassificationAndDistance : IComparable<ClassificationAndDistance>
    {
        public string Classification { get; }
        public double Similarity { get; }
        public int ClassificationNo { get; }

        public ClassificationAndDistance(string classification, double similarity, int classificationNo)
        {
            Classification = classification;
            Similarity = similarity;
            ClassificationNo = classificationNo;
        }

        public int CompareTo(ClassificationAndDistance other) => Similarity.CompareTo(other.Similarity);
    }

    public class KNeighborsClassifier<TIN, TOUT> : IClassifierKnn<TIN, TOUT>
    {
        private int _nNeighbors = 1;
        private DefaultDictionary<string, List<int[]>> _sdrMap = new DefaultDictionary<string, List<int[]>>();
        private int _sdrs = 10;

        public Dictionary<string, List<int[]>> StoredSDRs => _sdrMap;

        private double CosineSimilarity(int[] sdr1, int[] sdr2)
        {
            double dotProduct = 0;
            double magnitude1 = 0;
            double magnitude2 = 0;

            for (int i = 0; i < sdr1.Length; i++)
            {
                dotProduct += (double)sdr1[i] * sdr2[i];  // Ensure calculations use double
                magnitude1 += (double)sdr1[i] * sdr1[i];
                magnitude2 += (double)sdr2[i] * sdr2[i];
            }

            magnitude1 = Math.Sqrt(magnitude1);
            magnitude2 = Math.Sqrt(magnitude2);

            return (magnitude1 == 0 || magnitude2 == 0) ? 0 : dotProduct / (magnitude1 * magnitude2);
        }


        private List<ClassifierResult<string>> Voting(Dictionary<int, List<ClassificationAndDistance>> mapping, short howMany)
        {
            var votes = new DefaultDictionary<string, int>();
            var similarities = new Dictionary<string, double>();

            foreach (var key in _sdrMap.Keys)
                similarities[key] = 0;

            foreach (var coordinates in mapping.Values)
            {
                foreach (var classification in coordinates.Take(_nNeighbors))
                {
                    votes[classification.Classification]++;
                    similarities[classification.Classification] += classification.Similarity;
                }
            }

            return votes.OrderByDescending(v => v.Value)
                        .Select(v => new ClassifierResult<string>
                        {
                            PredictedInput = v.Key,
                            Similarity = similarities[v.Key],
                            PredictedLabel = v.Key,
                            PredictedSDRs = GeneratePredictedSDR(v.Key)
                        })
                        .Take(howMany)
                        .ToList();
        }

        private List<int> GeneratePredictedSDR(string label)
        {
            if (!_sdrMap.ContainsKey(label) || _sdrMap[label].Count == 0)
                return new List<int>();

            var referenceSDR = _sdrMap[label].OrderByDescending(sdr => sdr.Length).First();
            var predictedSDR = new HashSet<int>(referenceSDR);

            foreach (var sdr in _sdrMap[label])
            {
                var commonBits = sdr.Intersect(referenceSDR).ToArray();
                if (commonBits.Length > 0)
                {
                    predictedSDR.UnionWith(commonBits);
                }
            }

            return predictedSDR.Take(40).ToList();
        }

        public List<ClassifierResult<TIN>> GetPredictedInputValues(Cell[] unclassifiedCells, short howMany = 1)
        {
            if (unclassifiedCells.Length == 0)
                return new List<ClassifierResult<TIN>>();

            var unclassifiedSequence = unclassifiedCells.Select(idx => idx.Index).ToArray();
            var mappedElements = new DefaultDictionary<int, List<ClassificationAndDistance>>();

            foreach (var sdrList in _sdrMap)
            {
                foreach (var (sequence, idx) in sdrList.Value.WithIndex())
                {
                    var similarity = CosineSimilarity(sequence, unclassifiedSequence);
                    mappedElements[sequence.Length].Add(new ClassificationAndDistance(sdrList.Key, similarity, idx));
                }
            }

            return Voting(mappedElements, howMany) as List<ClassifierResult<TIN>>;
        }

        public void Learn(TIN input, Cell[] cells)
        {
            var label = input as string;
            int[] cellIndices = cells.Select(idx => idx.Index).ToArray();

            if (!_sdrMap[label].Exists(seq => cellIndices.SequenceEqual(seq)))
            {
                if (_sdrMap[label].Count > _sdrs)
                    _sdrMap[label].RemoveAt(0);
                _sdrMap[label].Add(cellIndices);
            }
        }

        public void ClearState() => _sdrMap.Clear();
    }
}
