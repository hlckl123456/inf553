Contains Jupyter notebook used to experiment with the Neural Network approach.

Input data path : Dataset/PA/Restaurants/

features_users_business.ipynb : Ipython notebook to compute feature vector for user and business

StackedNN.ipynb: 
It contains two parallel neural networks with two hidden layers each for user and business features. The user features and business features are modelled using the review vector created by LDA topic modelling.
It computes the RMSE for star rating. 
Number of user attr: 32
Number of business attr: 337


DeepNN_Attr.ipynb: 
It is a simple neural network with two hidden layers of 64 and 32 neurons each, it takes input the business feature vector and predicts the value of attribute based on the attribute you choose to train. Train Neural Net once for each of the attribute.
Here I have computed RMSE for the following attributes:
  - attributes.Ambience.casual	
  - attributes.BikeParking	
  - attributes.BusinessAcceptsCreditCards	
  - attributes.BusinessParking.street	
  - attributes.HasTV	
  - attributes.RestaurantsDelivery	
  - attributes.GoodForMeal.breakfast	
  - attributes.GoodForMeal.brunch	
  - attributes.GoodForMeal.dessert	
  - attributes.GoodForMeal.dinner	
  - attributes.GoodForMeal.latenight	
  - attributes.GoodForMeal.lunch	
  - attributes.Alcohol	
  - attributes.NoiseLevel	
  - attributes.PriceRange2

The attributes like Alcohol, Noise level have values more than 2 need to be encoded into one hot vector
 


