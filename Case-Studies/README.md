# Data Science Case Studies
The content of this folder comes from Chip Huyen's chapter on [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design/blob/master/build/build1/consolidated.pdf) for her upcoming book ["Machine Learning Interviews"](https://huyenchip.com/2019/07/21/machine-learning-interviews.html), which I highly recommend to subscribe.

Here are the corresponding [industry examples](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/Case-Studies/Industry-Examples.md) and [exercises](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/Case-Studies/Exercises.md) to work on.

Designing a machine learning system is an iterative process. There are generally four main components of the process: project setup, data pipeline, modeling (selecting, training, and debugging your model), and serving (testing, deploying, maintaining).

The output from one step might be used to update the previous steps. Some scenarios:
* After examining the available data, you realize it's impossible to get the data needed to solve the problem you previously defined, so you have to frame the problem differently.
* After training, you realize that you need more data or need to re-label your data.
* After serving your model to the initial users, you realize that the way they use your product is very different from the assumptions you made when training the model, so you have to update your model.

When asked to design a machine learning system, you need to consider all of these components.

## Project setup

You should first figure out as much detail about the problem as possible.

- **Goals**: What do you want to achieve with this problem? For example, if you're asked to create a system to rank what activities to show first in one's newsfeed on Facebook, some of the possible goals are: to minimize the spread of misinformation, to maximize revenue from sponsored content, or to maximize users' engagement.
- **User experience**: Ask your interviewer for a step by step walkthrough of how end users are supposed to use the system. If you're asked to predict what app a phone user wants to use next, you might want to know when and how the predictions are used. Do you only show predictions only when a user unlocks their phone or during the entire time they're on their phone?
- **Performance constraints**: How fast/good does the prediction have to be? What's more important: precision or recall? What's more costly: false negative or false positive? For example, if you build a system to predict whether someone is vulnerable to certain medical problems, your system must not have false negatives. However, if you build a system to predict what word a user will type next on their phone, it doesn't need to be perfect to provide value to users.
- **Evaluation**: How would you evaluate the performance of your system, during both training and inferencing? During inferencing, a system's performance might be inferred from users' reactions, e.g. how many times they choose the system's suggestions. If this metric isn't differentiable, you need another metric to use during training, e.g. the loss function to optimize. Evaluation can be very difficult for generative models. For example, if you're asked to build a dialogue system, how do you evaluate your system's responses?
- **Personalization**: How personalized does your model have to be? Do you need one model for all the users, for a group of users, or for each user individually? If you need multiple models, is it possible to train a base model on all the data and finetune it for each group or each user?
- **Project constraints**: These are the constraints that you have to worry about in the real world but less so during interviews: how much time you have until deployment, how much compute power is available, what kind of talents work on the project, what available systems can be used, etc.

## Data pipeline

Next you would like to know the details about the data pipelines:

- **Data availability and collection**: What kind of data is available? How much data do you already have? Is it annotated and if so, how good is the annotation? How expensive is it to get the data annotated? How many annotators do you need for each sample? How to resolve annotators' disagreements? What's their data budget? Can you utilize any of the weakly supervised or unsupervised methods to automatically create new annotated data from a small amount of humanly annotated data?
- **User data**: What data do you need from users? How do you collect it? How do you get users' feedback on the system, and if you want to use that feedback to improve the system online or periodically?
- **Storage**: Where is the data currently stored: on the cloud, local, or on the users' devices? How big is each sample? Does a sample fit into memory? What data structures are you planning on using for the data and what are their tradeoffs? How often does the new data come in?
- **Data preprocessing & representation**: How do you process the raw data into a form useful for your models? Will you have to do any featuring engineering or feature extraction? Does it need normalization? What to do with missing data? If there's class imbalance in the data, how do you plan on handling it? How to evaluate whether your train set and test set come from the same distribution, and what to do if they don't? If you have data of different types, say both texts, numbers, and images, how are you planning on combining them?
- **Challenges**: Handling user data requires extra care, as any of the many companies that have got into trouble for user data mishandling can tell you.
- **Privacy**: What privacy concerns do users have about their data? What anonymizing methods do you want to use on their data? Can you store users' data back to your servers or can only access their data on their devices?
- **Biases**: What biases might represent in the data? How would you correct the biases? Are your data and your annotation inclusive? Will your data reinforce current societal biases?

## Modeling

Modeling, including model selection, training, and debugging, is what's often covered in most machine learning courses. However, it's only a small component of the entire process. Some might even argue that it's the easiest component.

## Serving

Before serving your trained models to users, you need to think of experiments you need to run to make sure that your models meet all the constraints outlined in the problem setup. You need to think of what feedback you'd like to get from your users, whether to allow users to suggest better predictions, and from user reactions, how to defer whether your model does a good job.

- Training and serving aren't two isolated processes. Your model will continuously improve as you get more user feedback. Do you want to train your model online with each new data point? Do you need to personalize your model to each user? How often should you update your machine learning model?
- If it's a prediction model, you might want to measure your model's confidence with each prediction so that you can show only predictions that your model is confident about. You might also want to think about what to do in case of low confidence -- e.g. would you refer your user to a human specialist or collect more data from them?
- You should also think about how to run inferencing: on the user device or on the server and the tradeoffs between them. Inferencing on the user phone consumes the phone's memory and battery, and makes it harder for you to collect user feedback. Inferencing on the cloud increases the product latency, requires you to set up a server to process all user requests, and might scare away privacy-conscious users.
- And there's the question of interpretability. If your model predicts that someone shouldn't get a loan, that person deserves to know the reason why. You need to consider the performance/interpretability tradeoffs. Making a model more complex might increase its performance but make the results harder to interpret.
- For complex models with many different components, it's especially important to conduct ablation studies -- removing each component while keeping the rest -- to determine the efficiency of each component. You might find components whose removals don't significantly reduce the model's performance but significantly reduce its complexity.
- You also need to think about the potential biases and misuses of your model. Does it propagate any gender and racial biases from the data, and if so, how will you fix it? What happens if someone with malicious intent has access to your system?
