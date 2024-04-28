1) ML Model Selected (Random Forest):Detecting fraud in transactions is like finding a needle in a haystack. Most transactions are legitimate, but a few are fraudulent, making the data imbalanced. To teach our model to spot these rare fraudulent transactions, we need to balance the data. SMOTE helps us create more examples of fraudulent transactions, so our model can learn better. Additionally, we use class weights to tell the model to pay more attention to the minority class (fraudulent transactions) during training. This way, our Random Forest model becomes more sensitive to detecting fraud while still being accurate overall."
2) Features Selected:I did Univariate analysis using heat map for checking the relation of feature also used hypotheses testings but finally I decided to use all available features in our dataset for predicting fraud. This ensures that we don't miss out on any potentially important information that could help our model make accurate predictions. By including all features, we give our model the best chance to identify patterns and relationships that might not be obvious from individual features alone. Additionally, this approach helps us avoid introducing biases or overlooking important information that could impact the model's performance.
3) Feature Importance : Please find the below barh plot for feature importance analysis.
   ![image](https://github.com/sagar0930/FraudTransaction/assets/103502762/88083cd3-bcae-4bd3-a939-a5ba42cdfaa7)


5) 4)Evaluation of model - In evaluating our Random Forest model for fraud detection, we focused on several key metrics to assess its performance. We looked at metrics such as accuracy, precision, recall, and F1-score to understand how well the model identifies both fraudulent and Non-fraudulent transactions.Please find the final result of trained module on test data set.
   
   

