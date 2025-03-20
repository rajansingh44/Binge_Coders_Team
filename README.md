

# Unit Test cases
### CosineSimilarity_IdenticalVectors_ReturnsOne
- *Test Category*: CosineSimilarityIdenticalVectors
- *Description*:  Verifies that the CosineSimilarity function returns 1.0 when comparing two identical vectors.

### CosineSimilarity_CompletelyDifferentVectors_ReturnsZero
- *Test Category*: CosineSimilarityDifferentVectors
- *Description*: Confirms that CosineSimilarity correctly returns 0.0 when given two completely dissimilar vectors.

### CosineSimilarity_PartiallySimilarVectors_ReturnsExpectedValue
- *Test Category*: CosineSimilarityPartialMatch
- *Description*: Ensures that the function returns a similarity score between 0.0 and 1.0 when given two partially similar vectors.


### ConvertPermanenceToBinary_ValidInput_ReturnsExpectedBinaryArray
- *Test Category*: PermanenceConversionValidInput
- *Description*: Ensures that the ConvertPermanenceToBinary function correctly converts permanence values into binary format using a 0.7 threshold.

### ConvertPermanenceToBinary_AllPositivePermanences_ReturnsExpectedValues
- *Test Category*: PermanenceConversionAllPositive
- *Description*: Verifies that the function correctly processes an array where all permanence values are non-negative.

### ConvertPermanenceToBinary_NegativeValue_ThrowsException
- *Test Category*: PermanenceConversionNegativeValues
- *Description*: Confirms that an ArgumentException is thrown when a permanence array contains negative values.


### AddActiveColsToStoredSDRs_StoresActiveColumnsCorrectly
- *Test Category*: SDRStorageValidation
- *Description*: Verifies that the AddActiveColsToStoredSDRs function correctly stores active column SDRs in a dictionary, ensuring accurate image reconstruction and classification.
