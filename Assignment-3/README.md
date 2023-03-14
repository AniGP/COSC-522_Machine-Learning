#### Assignment 3: Motion-based Activity Recognition

- The goal is to develop a machine learning pipeline to recognize different activities using a phone’s motion sensors (accelerometer, gyroscope). The activities we are looking for are:

1. Walking
2. Climbing up the stairs
3. Climbing down the stairs
4. Standing up
5. An activity of your choice

- Building a model that detects these activities will require quite a bit of data and it will be impractical to rely on the usual ~2 week period between assignments to collect data, clean it, extract features, and build a model. Thus, we are assigning the assignment 4 sooner so you can start the data collection right away and perform other steps of the pipeline later.

- In any machine learning task, it is often hard to predict how much training data will be useful. Ideally, you want to collect some pilot data, look at how the models converge to estimate how much more data might be needed. This can be time-consuming and for the purpose of this assignment, we are providing some guideline for the amount of data (treat these numbers as lower bounds as more data is not going to hurt):

walking: 4 hours
climbing up the stairs: 20 minutes
climbing down the stairs: 20 minutes
standing up: 50 instances
an activity of your choice (up to you).

- For the actual data collection, you can use existing apps available for different mobile OS. For Android, I would recommend using AndroSensor app and for iOS, use SensorLog. I have personally tested these apps and they work reasonably well. There are plenty of other apps that might work too. Feel free to use any of those if you like. None of these apps (including my suggestions) provide a way to annotate your data in the app. You will need to develop strategies to keep a log of the actual activity being performed. I would recommend to record only one activity in one session and keep the files separate.

- I used SensorLog on my iOS device to capture the data.

- As you start collecting some initial data, read the files into Python, plot the data, and make sure it makes sense. As we discuss various parts of the machine learning pipeline in the course, keep adding those modules to your code to make sure everything seems reasonable. Then as the semester progresses, keep adding data, and keep adding new capabilities to the pipeline.