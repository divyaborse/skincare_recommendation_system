"# skincare_recommendation_system" 
Four types of skincare recommendation systems are as follows:
1)Product recommendation based on user skin features
2)Product recommendation based on user favorites
3)Product recommendation based on brands
4)product recommendation based on product category
Collaborative filtering:
We have used user-based collaborative filtering  which predicts the items that users might like based on the rating given by other users who share similar taste with the target user.
In our case, if one user shares skin type, tone, eye color, hair color, with another user, there is a chance they will enjoy the same products.

Content-based filtering:
we have applied content based recommendation for two reason: 
Actually it is inappropriate to predict the beauty products based on user’s past purchase history.That’s because past products are much smaller than the test data.Predicting products based on past experience becomes difficult and does not give guarantee result because user must have tried some cosmetics but number of cosmetic products in the world are much larger than that.
There could be some people who have very similar taste.So here we can use user-user collaborative filtering to recommend new cosmetic products based on ranking values on its neighbouring groups.But every person has different skin type and feature so it’s become a very ticky problem  to recommend the right  product to the user. So to get reliability and stability in the recommendation we need to focus on ingredients  of each product and get similarities based on them.

If user enters skin features then based on his/her skin features ,beauty products will be recommended to the user.If user enters brand name then products associated with that brand will be recommended to the user.If user enters one of his favorite products then products which share similarity in terms of targeted products ingredients will be recommended to the user
If user enters product category then user will get all products belonging to that category.

################**How to run**################
run command :
cd demoapp
cd demoapp
python testfilepython.py


