# Capstone: Air Bnb Listing Analysis of Paris & London

### Introduction: 

With the majority of work becoming remote, this opens up the opportunity for working in any environment with a quiet corner and dependable Wifi. In fact, one of the draws I find for working in the tech industry is the freedom of melding travel and work together. AirBnb's platform is increasingly becoming an avenue of not just vacation rentals, but short term accomodation to give guests an opportunity live and work in a new city or new part of the world for more than a long weekend. 

With this being said, I wanted to prepare some findings on what features affect the pricing of an Air Bnb listing to help potential hosts increase the value or appeal of their listing.

Since this is my capstone, I want to deep dive into a specific skill and give a thorough analysis basing my decisions off EDA and statistical reasoning. As a result, I will deep dive into pricing predictions because I enjoyed doing price predictions with the Ames, IA dataset; however, I believe some of my assumptions during EDA and modeling were ancedotal, so I would like to revisit the concept to see how much progress I've made since this project. 
   
### Problem Statement: 

 - What are the features of a listing which contribute most to the pricing? 
 
Short answer: 
Room type more specifically private room, how many people the listing accommodates, & neighborhood

Long answer: 
According to the correlation of features, the top five factors which contribute most to predicting price are how many people the Air Bnb accommodates, bedroom count, availability in three different windows of time and bathroom count sandwiched in the middle of availability windows. In addition to looking at correlation, I used a tool from SckitLearn called SelectKBest which pulls out the best features for a regression model to use, the features selected by this tool were similar to the highly correlated features with some exceptions like host id and calculated host listing count.  

With varying levels of weight, the models both equate private room as being one of the most important features affecting price. Then XGBRegressor lists hotel room, shared room, and how many people the unit accommodates as the next three most important features affecting the listing price. While RandomForestRegressor lists accommodates, availability of the unit within a year's time, and the latitude/longitude of the unit as the most important features affecting price. Although I chose XGBRegressor as my final model based off the metrics, I find the feature importance of RandomForestRegressor to be a much more intuitive bundle of results, but I digress. 

Based off these two perspectives, my final insights to answer my problem statement are suggesting if available - renting out a spare room may be the best piece of advice I could give to someone looking to break into the AirBnb market because hosts can charge around ~58 USD per night. Furthermore, if a host has accommodations for two guests, they can charge around ~80 USD per night. 

In addition to these features, the London neighborhoods appear higher in the feature importance lists so I can only imagine having a listing in neighborhoods like Westminster and Kensington & Chelsea drive the price as well. The features, in a loose sense of the term, have a macro focus - location of listing, room type, and how many people the unit accommodates are the important features affecting price. The best a host could hope for would be a private room in Westminster because one is able to charge nearly 44% more per night. 

### Data Chosen: 

 - [`london_airbnb_jun22.csv`](../data/london_airbnb_jun22.csv): June 2022 Inside Airbnb London Listings
 - [`london_airbnb_mar22.csv`](../data/london_airbnb_mar22.csv): March 2022 Inside Airbnb London Listings
 - [`london_airbnb_dec21.csv`](../data/london_airbnb_dec21.csv): December 2021 Inside Airbnb London Listings
 - [`london_airbnb_sept21.csv`](../data/london_airbnb_sept21.csv): September 2021 Inside Airbnb London Listings
 - [`paris_airbnb_jun22.csv`](../data/paris_airbnb_jun22.csv): June 2022 Inside Airbnb Paris Listings
 - [`paris_airbnb_mar22.csv`](../data/paris_airbnb_mar22.csv): March 2022 Inside Airbnb Paris Listings
 - [`paris_airbnb_dec21.csv`](../data/paris_airbnb_dec21.csv): December 2021 Inside Airbnb Paris Listings
 - [`paris_airbnb_sept21.csv`](../data/paris_airbnb_sept21.csv): September 2021 Inside Airbnb Paris Listings
 
### Data Dictionary: 

|Feature|Type|Dataset|Description|
|---|---|---|---|
|city|int64|bnb|Identifies city of Airbnb listing binarized 0: Paris, 1: London (feature engineered)|
|host_id|int64|bnb|Airbnb's unique identifier for the host/user| 
|superhost|int64|bnb|Identifies if host is a superhost binarized 0: False, 1: True|
|neighborhood|object|bnb|The neighbourhood as geocoded using the latitude and longitude against neighborhoods as defined by open or public digital shapefiles.|
|latitude|float64|bnb|Uses the World Geodetic System (WGS84) projection for latitude and longitude.|
|longitude|float64|bnb|Uses the World Geodetic System (WGS84) projection for latitude and longitude.|
|property_type|object|bnb|Self selected property type. Hotels and Bed and Breakfasts are described as such by their hosts in this field.|
|room_type|object|bnb|Listing types grouped into three different types: Entire place, private room, & shared room|
|accomodates|int64|bnb|The maximum capacity of the listing|
|baths|float64|bnb|The number of bathrooms in the listing|
|beds|float64|bnb|The number of bedrooms in the listing|
|price|float64|bnb|Daily price in local currency converted to USD|
|minimum_nights|int64|bnb|minimum number of night stay for the listing (calendar rules may be different)|
|maximum_nights|int64|bnb|maximum number of night stay for the listing (calendar rules may be different)|
|minimum_minimum_nights|float64|bnb|the smallest minimum_night value from the calender (looking 365 nights in the future)|
|maximum_minimum_nights|float64|bnb|the largest minimum_night value from the calender (looking 365 nights in the future)|
|minimum_maximum_nights|float64|bnb|the smallest maximum_night value from the calender (looking 365 nights in the future)|
|maximum_maximum_nights|float64|bnb|the largest maximum_night value from the calender (looking 365 nights in the future)|
|minimum_nights_avg_ntm|float64|bnb|the average minimum_night value from the calender (looking 365 nights in the future)| 
|maximum_nights_avg_ntm|float64|bnb|the average maximum_night value from the calender (looking 365 nights in the future)|
|has_availability|int64|bnb|Listing availability binarized 0: False, 1: True|
|availability_30|int64|bnb|The availability of the listing 30 days in the future as determined by the calendar.|
|availability_60|int64|bnb|The availability of the listing 60 days in the future as determined by the calendar.|
|availability_90|int64|bnb|The availability of the listing 90 days in the future as determined by the calendar.|
|availability_365|int64|bnb|The availability of the listing 365 days in the future as determined by the calendar.|
|review_scores_rating|float64|bnb|Overall listing rating scaled 0-5|
|review_scores_accuracy|float64|bnb|Accuracy of amenities scaled 0-5|
|review_scores_cleanliness|float64|bnb|Cleanliness of listing scaled 0-5|
|review_scores_checkin|float64|bnb|Check-in Process rating scaled 0-5|
|review_scores_communication|float64|bnb|Host communication effectiveness scaled 0-5|
|review_scores_location|float64|bnb|Location rating scaled 0-5|
|review_scores_value|float64|bnb|Overall value of listing scaled 0-5|
|calculated_host_listings_count|int64|bnb|The number of listings the host has in the current scrape, in the city/region geography|
|calculated_host_listings_count_entire_homes|int64|bnb|The number of Entire home/apt listings the host has in the current scrape, in the city/region geography|  
|calculated_host_listings_count_private_rooms|int64|bnb|The number of Private room listings the host has in the current scrape, in the city/region geography| 
|calculated_host_listings_count_shared_rooms|int64|bnb|The number of Shared room listings the host has in the current scrape, in the city/region geography|

<a href="https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit#gid=982310896">(Inside Airbnb Data Dictionary)</a>

### Final Thoughts

Overall, the models performed quite well given the size of the data and the fact two completely different internationally renowned destinations were incorporated. When running initial modeling tests to ensure I would be able to use this dataset, a basic linear regression model was scoring at .21 R2 score on just the numeric data with all nulls dropped. I have to say I am proud of my thorough approach and appreciative of my instructors for guiding me through this process. The greatest helper to the modeling phase was in my cleaning notebook were I used the IQR to address outliers which caused the correlation of my top features to double. In addition, my due diligence on the reason behind dropping 10% of null values in price which turned out to be rental units in Paris was a win as well.

My goal in selecting a final model between the two top performing models was to ensure the model was able to pass all LINE assumptions. The LINE assumptions are L-Linearity, I-Independent, N-Normal, E-Equal Variance. From the residual scatter plots of our two models, we see a clear violation of the last LINE assumption which is equal variance. In order for our models to pass all this assumption, there must be no pattern and the residual, or error term is constant - homoscedasticity. 

Even if the other LINE assumptions are being followed such as the residual plots being normally distributed, each event occuring independent of the other, and our target variable, price, having a linear relationship with the x-variables, or features. Again, as I have said before, my problem statement is based around how features affect price and will need to pull out feature importances to interpret their relationship to price. With this being said, our insights and interpretations should be given with a word of caution since one of the LINE assumptions has been violated. 

During some discussion with my instructors, we arrived at the potential reason for this diamond pattern may be due to categorical features. Luckily, my dataset contains only a couple categorical features, neighborhood & room_type. The next step I will take in diving deeper into why this violation occured is by taking a subset of the largest neighborhood out of the two cities I am analyzing. I will then repeat the exact same modeling steps, scoring, and plotting to see if zooming in on a specific neighborhood will allow me to gain better insights.  

Since I have run the subset modeling phase and again find myself in nearly the same boat, I will proceed with giving insights to help solve my problem statement, but I will be giving them with a fair announcement of caution. The scoring and plots followed a similar pattern; however, there was an improvement with the scatter plots becoming a bit more equally varied. 

An update I received during some more discussion was surrounding the LINE assumptions only being applicable to linear models and not useful in tree based models which are the two top performing models I have chosen to pull feature importances out and optimize. This is reassuring because I was unsure how to proceed after examining my data on neighborhood level for the residuals to generate a similar pattern. I was given some guidance on using another metric to give my audience easily explainable insight through the MAE - mean absolute error. As a result, I will also be using the MAE for help in chosing a final model. 

When looking at the mean absolute error, XGBRegressor, MAE of 2.24 USD, does around 100x better at predicting the price of listings compared to RandomForestRegressor, MAE of 22.97 USD. Furthermore, when I zoom the lens in closer to a neighborhood level, the XGBRegressor can predict pricing within 0.16 cents and RandomForestRegressor can predict pricing within 17 USD. Even though my final decision was XGBRegressor as my top performing model , I was interested in seeing the two different perspectives of feature importances from my two top performing models to give more robust insights to AirBnb hosts on what drives the price most.