# transformer_chatbot
Transformer based ChatBot--

Our Approach-

the codebase is an archive of a proposed solution to INEURON'S AI hackathon.
The problem statement being " build an intelligent virtual agent for answering queries raised by students on chat"
the preprocessing consisted of extracting the question -answer pairs from the json provided.

#### We noticed that what "constitutes" a chat can be a sequence of messages from either the student or the replying human .



The hyperparameters are adjusted to suit the computational restrictionns of the local systems.

Post training, the model was deployed as an Endpoint using Flask and a basic UI was used to expose the API for hitting and getting answers to queries.


Taking care of business and user experience we added a feature of "assisted video links" in the bot reply when and where required.
#### Also we took care of latency issue, even though it is a transformer based model , latecny of response ~<2secs

# Packages-used------------
tensorflow==2.8.2

pandas==1.3.5

numpy==1.21.6

tensorflow-datasets

beautifulsoup4

Flask

google

# Model-architecture----------------------------------
Post cleaning and experimenting with few of the architectures/algorithms we trained a "TEXT GENERATION MODEL"
using transformers . The architecture does not uses any pretrained weights , rather its trained from scratch.

![Screenshot 2022-06-26 122440](https://user-images.githubusercontent.com/70206828/175803152-9be69e5f-0de3-4b10-95b9-81e0e01f48c0.png)




![new_ModalNet-21](https://user-images.githubusercontent.com/70206828/175798243-32f9a7c0-d2c0-41ba-b5e6-a23219651e22.jpg)
