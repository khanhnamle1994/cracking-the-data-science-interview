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



4. [How we grew from 0 to 4 million women on our fashion app, with a vertical machine learning approach](https://medium.com/hackernoon/how-we-grew-from-0-to-4-million-women-on-our-fashion-app-with-a-vertical-machine-learning-approach-f8b7fc0a89d7) (Gabriel Aldamiz, HackerNoon, 2018)



5. [Machine Learning-Powered Search Ranking of Airbnb Experiences](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789) (Mihajlo Grbovic, Airbnb Engineering & Data Science, 2019)



6. [From shallow to deep learning in fraud](https://eng.lyft.com/from-shallow-to-deep-learning-in-fraud-9dafcbcef743) (Hao Yi Ong, Lyft Engineering, 2018)



7. [Space, Time and Groceries](https://tech.instacart.com/space-time-and-groceries-a315925acf3a) (Jeremy Stanley, Tech at Instacart, 2017)



8. [Uber's Big Data Platform: 100+ Petabytes with Minute Latency](https://eng.uber.com/uber-big-data-platform/) (Reza Shiftehfar, Uber Engineering, 2018)



9. [Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning](https://blogs.dropbox.com/tech/2017/04/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning/) (Brad Neuberg, Dropbox Engineering, 2017)



10. [Scaling Machine Learning at Uber with Michelangelo](https://eng.uber.com/scaling-michelangelo/) (Jeremy Hermann and Mike Del Balso, Uber Engineering, 2019)
