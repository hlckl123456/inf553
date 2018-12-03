Include /PA in this directory if you wish to run the script

File: memoryBasedCF.py
This Python script uses review text build a review corpus and compute user-user similarities with
cosine similarity on tf-idf user-review vectors of 500 features. The script then finds the nearest 10 users
for each user in review and computes an average business id vector to be used as prediction results.

The output is a result folder which contains:
- Predictions for train
- Predictions for validation
- Predictions for test
- Validation user predictions and their top 50 similar business
- Test user predictions and their top 50 similar business
- Validation hit ratio
- Test hit ratio
- Train RMSE vector for 16 business attributes
- Validation RMSE vector for 16 business attributes
- Test RMSE vector for 16 business attributes

To run this script call 
$python memoryBasedCF.py
