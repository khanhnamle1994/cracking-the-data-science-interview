Here are the list of industry examples that Chip compiles:

1. [Using Machine Learning to Predict Value of Homes On Airbnb](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d) (Robert Chang, Airbnb Engineering & Data Science, 2017)

**Problem**: Predict Customer Lifetime Value of new listings

**ML Workflow**:

- Feature Engineering: Define relevant features using Airbnb's internal feature repository Zipline.
- Prototyping and Training: Train a model prototype using scikit-learn.
- Model Selection and Validation: Perform model selection and validation using various AutoML frameworks.
- Productionization: Take the selected model prototype to production using Airbnb's notebook translation framework ML Automator into an Airflow ML pipeline.

2. [Using Machine Learning to Improve Streaming Quality at Netflix](https://medium.com/netflix-techblog/using-machine-learning-to-improve-streaming-quality-at-netflix-9651263ef09f) (Chaitanya Ekanadham, Netflix Technology Blog, 2018)

**Problem**: Use ML to improve streaming quality and adapt to different/fluctuating conditions (networks and devices with widely varying capabilities)

- **Network Quality Characterization and Prediction**:
	- Can we predict what throughput will look like in the next 15 minutes given the last 15 minutes of data?
	- How can we incorporate longer-term historical information about the network and device?
	- What kind of data can we provide from the server that would allow the device to adapt optimally?
	- Even if we cannot predict exactly when a network drop will happen, can we at least characterize the distribution of throughput that we expect to see given historical data?
- **Video Quality Adaptation During Playback**:
	- Can we leverage data to determine the video quality that will optimize the quality of experience?
	- The quality of experience can be measured in several ways, including the initial amount of time spent waiting for video to play, the overall video quality experienced by the user, the number of times playback paused to load more video into the buffer ("rebuffer"), and the amount of perceptible fluctuation in quality during playback.
	- These metrics can trade off with one another. This "credit assignment" problem is a well-known challenge when learning optimal control algorithms, and ML techniques have great potential to tackle these issues.
- **Predictive Caching**:
	- Predicting what a user will play in order to cache it on the device before the user hits play, enabling the video to start faster and/or at a higher quality.
	- By combining various aspects of their viewing history together with recent user interactions and other contextual variables, one can formulate this as a supervised learning problem where we want to maximize the model's likelihood of caching what the user actually ended up playing, while restricting constraints around resource usage coming from the cache size and available bandwidth.
- **Device Anomaly Detection**:
	- Netflix has history on alerts that were triggered as well as the ultimate determination of whether or not device issues were in fact real and actionable. The data can then be used to train a model that can predict the likelihood that a given set of measured conditions constitutes a real problem.
	- It's challenging to determine the root cause of a problematic issue. Statistical modeling can help by controlling for various covariates.

3. [150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com](https://blog.acolyer.org/2019/10/07/150-successful-machine-learning-models/) (Bernardi et al., KDD, 2019).

- **Different types of model**: The models deployed at Booking.com can be grouped into six broad categories:
	- **Traveller preferences models** operate in the semantic layer, and make broad predictions about user preferences (e.g., degree of flexibiilty).
	- **Traveller context models**, also semantic, which predictions about the context in which a trip is taking place (e.g. with family, with friends, for business, …).
	- **Item space navigation models** which track what a user browses to inform recommendations both the the user’s history and the catalog as a whole.
	- **User interface optimisation models** optimise elements of the UI such as background images, font sizes, buttons etc. Interestingly here, “we found that it is hardly the case that one specific value is optimal across the board, so our models consider context and user information to decide the best user interface.”
	- **Content curation models** curate human-generated content such as reviews to decide which ones to show
	- **Content augmentation models** compute additional information about elements of a trip, such as which options are currently great value, or how prices in an area are trending.

- **Lesson 1: projects introducing machine learned models deliver strong business value**
	- All of these families of models have provided business value at Booking.com. Moreover, compared to other successful projects that have been deployed but did not use machine learning, the machine learning based projects tend to deliver higher returns.
	- Once deployed, beyond the immediate business benefit they often go on to become a foundation for further product development.
- **Lesson 2: model performance is not the same as business performance**
	- An interesting finding is that increasing the performance of a model does not necessarily translate into a gain in [business] value.
	- This could be for a number of reasons including saturation of business value (there’s no more to extract, whatever you do); segment saturation due to smaller populations being exposed to a treatment (as the old and new models are largely in agreement); over-optimisation on a proxy metric (e.g. clicks) that fails to convert into the desired business metric (e.g. conversion); and the uncanny valley effect.
- **Lesson 3: be clear about the problem you’re trying to solve**
	- Before you start building models, it’s worth spending time carefully constructing a definition of the problem you are trying to solve. The Problem Construction Process takes as input a business case or concept and outputs a well-defined modeling problem (usually a supervised machine learning problem), such that a good solution effectively models the given business case or concept.
	- Some of the most powerful improvements come not from improving a model in the context of a given setup, but changing the setup itself. For example, changing a user preference model based on click data to a natural language processing problem based on guest review data.
- **Lesson 4: prediction serving latency matters**
	- In a experiment introducing synthetic latency, Booking.com found that an increase of about 30% in latency cost about 0.5% in conversion rates “a relevant cost for our business“. This is particularly relevant for machine learned models since they require significant computational resources when making predictions. Even mathematically simple models have the potential of introducing relevant latency.
	- Booking.com go to some lengths to minimise the latency introduced by models, including horizontally scaled distributed copies of models, a in-house developed custom linear prediction engine, favouring models with fewer parameters, batching requests, and pre-computation and/or caching.
- **Lesson 5: get early feedback on model quality**
	- When models are serving requests, it is crucial to monitor the quality of their output but this poses at least two challenges: (1) Incomplete feedback due to the difficulty of observing true labels; and (2) Delayed feedback e.g. a prediction made at time of booking as to whether a user will leave a review cannot be assessed until after the trip has been made.
	- One tactic Booking.com have successfully deployed in these situations with respect to binary classifiers is to look at the distribution of responses generated by the model.
- **Lesson 6: test the business impact of your models through randomised controlled trials**
	- The paper includes suggestions for how to set up the experiments under different circumstances.
	- When not all subjects are eligible to be exposed to a change (e.g., they don’t have a feature the model requires), create treatment and non-treatments groups from within the eligible subset.
	- If the model only produces outputs that influence the user experience in a subset of cases, then further restrict the treatment and non-treatment groups to only those cases where the model produces a user-observable output (which won’t of course be seen in the non-treatment group). To assess the impact of performance add a third control group where the model is not invoked at all.
	- When comparing models we are interested in situations where the two models disagree, and we use as a baseline a control group that invokes the current model (assuming we’re testing a current model against a candidate improvement).

4. [How we grew from 0 to 4 million women on our fashion app, with a vertical machine learning approach](https://medium.com/hackernoon/how-we-grew-from-0-to-4-million-women-on-our-fashion-app-with-a-vertical-machine-learning-approach-f8b7fc0a89d7) (Gabriel Aldamiz, HackerNoon, 2018)

- **Thesis**: Outfits are the best asset to understand people’s taste. Understanding taste will transform online fashion. The Fashion Taste API builds a Taste Graph for each fashion retailer, their system of intelligence to understand why a shopper buys a product. It’s a strategic asset that includes Taste Profiles, a fashion ontology, and our own interpretation of fashion products.
- **1st Step: Building the app for people to express their needs** - Three things helped with retention:
	- (a) identify retention levers using behavioral cohorts.
	- (b) re-think the onboarding process, once we knew the levers of retention.
	- (c) define how we learn.
- **2nd step: Building the data platform to learn people’s fashion needs**
	- Social Fashion Graph is a compact representation of how needs, outfits and people interrelate, a concept that helped us build the data platform. The data platform creates a high-quality dataset linked to a learning and training world, our app, which therefore improves with each new expression of taste.
	- We ended up opening up this technology to fashion retailers, with the Fashion Taste API. The objective of the Fashion Taste API is to empower teams to own taste data from each of their shoppers, so they can built memorable experiences for their clients. Truly effective fitting room smart mirrors, in-store fashion stylists, and personalized omnichannel experiences.
	- We thought of outfits as playlists: an outfit is a combination of items that makes sense to consume together. Using collaborative filtering, the relations captured here allow us to offer recommendations in different areas of the app.
- **3rd Step: Algorithms**
	- Fashion has its own challenges.
	- There is not an easy way to match an outfit to a shoppable product (think about most garments in your wardrobe, most likely you won’t find a link to view/buy those garments online, something you can do for many other products you have at home). Another challenge: the industry is not capturing how people describe clothes or outfits, so there is a strong disconnect between many ecommerces and its shoppers. Another challenge: style is complex to capture and classify by a machine.
	- Owning the correct data set allows us to focus on the specific narrow use cases related to outfit recommendations, and to focus on delivering value through the algorithms instead of spending time collecting and cleaning data. People’s very personal style can become as actionable as metadata and possibly as transparent as well (?), and I think we can see the path to get there. As we have a consumer product that people already love, we can ship early results of these algorithms partially hidden, and increase their presence as feedback improves results.

5. [Machine Learning-Powered Search Ranking of Airbnb Experiences](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789) (Mihajlo Grbovic, Airbnb Engineering & Data Science, 2019)

**Problem**: Building a Search Ranking for Airbnb Experiences

- **Stage 1: Build a Strong Baseline**
	- *Collect training data*: Collected search logs / clicks of users who ended up making bookings.
	- *Label training data*: 2 labels (experiences that were booked + experiences that were clicked but not booked)
	- *Build signals based on which to be ranked*: Built 25 experience features.
	- *Train the ranking model*: Given the training data, labels, and features, used Gradient Boosted Decision Tree model. Treated the problem as binary classification with log-loss loss function.
	- *Test the ranking model*:
		- To perform offline hyper-parameter tuning and comparison to random re-ranking in production, used hold-out data which was not used in training. Choice of metrics were AUC and NDCG.
		- Plotted partial dependency plots for several most important Experience features -> showing what would happen to specific Experience ranking scores if all values but a single feature are fixed.
		- Conducted an online A/B experiment between proposed model and rule-based random ranking in terms of number of bookings -> Improve bookings by +13%.
	- *Implementation Details*: The entire ranking pipeline, including training and scoring, was implemented offline and ran daily in Airflow. The output was just a complete ordering of all Experiences (an ordered list).

- **Stage 2: Personalize**
	- *Personalize based on Booked Airbnb Homes*: Use 2 features, (1) Distance between Booked Home and Experience; and (2) Experience available during Booked Trip.
	- *Personalize based on User's Clicks*: Compute 2 features derived from user clicks and categories of clicked Experiences: (1) Category Intensity; and (2) Category Recency.
	- *Training the ranking model*:
		- Generated training data that contains Personalization features by reconstructing the past based on search logs.
		- Train 2 models: one with personalization features for logged-in users and one without personalization features that will serve log-out traffic.
	- *Test the ranking model*: Conducted A/B tests to compare the new setup with 2 models with Personalization features to the model from Stage 1 -> Improve bookings by +7.9%.

- **Stage 3: Move to Online Scoring**
	- Used Query Features and user's browser language setting / country information.
	- *Training the ranking model*: Trained 2 GBDT models
		- Model for logged-in users, which uses Experience Features, Query Features, and User (Personalization) Features.
		- Model for logged-out traffic, which uses Experience and Query Features, trained using data of logged-in users but not considering Personalization Features.
	- *Test the ranking model*: Conducted an A/B test to compare the Stage 3 to Stage 2 models -> Improve bookings by +5.1%.

- **Stage 4: Handle Business Rules**
	- Promote quality.
	- Discover and promote potential new hits early.
	- Enforce diversity in the top 8 results.
	- Optimize Search without Location for Click-ability.

- **Monitor and Explain Rankings**
	- Give hosts concrete feedback on what factors lead to improvement in the ranking and what factors lead to decline.
	- Keep track of the general trends that the ranking algorithm is enforcing to make sure it is the behavior for Airbnb's marketplace.
	- Used Apache Superset and Airflow to create 2 dashboards: (1) Dashboard that tracks rankings of specific Experiences in their market over time, as well as values of feature used by the ML model; and (2) Dashboard that shows overall ranking trends for different groups of Experiences (e.g. how 5-star Experiences rank in their market).

- **Ongoing and Future Work**
	- Training data construction (by logging the feature values at the time of scoring instead of reconstructing them based on best guess for that day)
	- Loss function (e.g. by using pairwise loss, where we compare booked Experience with Experience that was ranked higher but not booked, a setup that is far more appropriate for ranking)
	- Training labels (e.g. by using utilities instead of binary labels, i.e. assigning different values to different user actions, such as: 0 for impression, 0.1 for click, 0.2 for click with selected date & time, 1.0 for booking, 1.2 for high quality booking)
	- Adding more real-time signals (e.g. being able to personalize based on immediate user activities, i.e. clicks that happened 10 minutes ago instead of 1 day ago)
	- Explicitly asking users about types of activities they wish to do on their trip (so we can personalize based on declared interest in addition to inferred ones)
	- Tackling position bias that is present in the data we use in training
	- Optimizing for additional secondary objectives, such as helping hosts who host less often than others (e.g. 1–2 a month) and hosts who go on vacation and come back
	- Testing different models beyond GBDT
	- Finding more types of supply that work well in certain markets by leveraging predictions of the ranking model.
	- Explore/Exploit framework
	- Test human-in-the-loop approach (e.g. Staff picks)

6. [From shallow to deep learning in fraud](https://eng.lyft.com/from-shallow-to-deep-learning-in-fraud-9dafcbcef743) (Hao Yi Ong, Lyft Engineering, 2018)

- How to improve Lyft's ML infrastructure?
- Logistic Regression to Gradient-Boosted Decision Trees.
- **Road to Production**
    - A simple and reliable way to serialize a prototype model on a Jupyter Notebook and load it onto a production system.
    - Built a library that utilizes the standardized scikit-learn API and pickle-based serialization
    - Use Tensorflow deep learning models in production because of their performance and their ability to work with signals that are hard to engineer features from.
    - Developing a more modern, container-based model execution to put a seamless prototype-to-production ML process within reach.
- **Climbing Over Walls**
    - Having a research scientist with good system knowledge helped guide and accelerate the engineering development process.
    - Additionally, the change in work scope reduced miscommunication between different roles and freed up engineers from rote feature implementation to focus more on, appropriately, building better platforms.

7. [Space, Time and Groceries](https://tech.instacart.com/space-time-and-groceries-a315925acf3a) (Jeremy Stanley, Tech at Instacart, 2017)

**Problem**: Logistics at Instacart => Stochastic Capacitated Vehicle Routing Problem with Time Windows for Multiple Trips

**ML System**:

- Predict the distribution of time expected for any given shopper and assignment.
- Decompose into sub-problems and solve them to near optimality.
- Apply heuristics for limiting search spaces, deal with anomalies, fine-tune solutions, and adapt under uncertainty.
- Recompute batch plans every minute and make dispatch decisions just in time.

**Use Visualization To**:

- Build intuition for how logistics system functions at scale.
- Generate hypotheses for ways to improve algorithms or operations.
- Confirm that changes to production have the expected behavior.
- Identify patterns that fast shoppers exhibit, and share those insights with other shoppers.
- Make better operational decisions about parking spaces, store locations, and product offering.

8. [Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning](https://blogs.dropbox.com/tech/2017/04/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning/) (Brad Neuberg, Dropbox Engineering, 2017)



9. [Scaling Machine Learning at Uber with Michelangelo](https://eng.uber.com/scaling-michelangelo/) (Jeremy Hermann and Mike Del Balso, Uber Engineering, 2019)

**ML Use Cases at Uber**
- Uber Eats
- Marketplace Forecasting
- Customer Support
- Ride Check
- Estimated Times of Arrivals
- One-Click Chat
- Self-Driving Cars

**How To Scale ML at Uber**
- Organization:
	- Product Teams
	- Specialist Teams
	- Research Teams
	- ML Platform Teams
- Process:
	- Launching Models
	- Coordinated Planning Across ML Teams
	- Community
	- Education
- Technology:
	- End-To-End Workflow
	- ML as Software Engineering
	- Model Developer Velocity
	- Modularity and Tiered Architecture
