# transformer_chatbot
Transformer based ChatBot

the codebase is an archive of a proposed solution to INEURON'S AI hackathon.
The problem statement being " build an intelligent virtual agent for answering queries raised by students on chat"
the preprocessing consisted of extracting the question -answer pairs fromm the json provided.

Post cleaning and experimenting with few of the architectures/algorithms we trained a "TEXT GENERATION MODEL"
using transformers . The architecture does not uses any pretrained weights , rather its trained from scratch.

The hyperparameters are adjusted to suit the computational restrictionns of the local systems.

Post training, the model was deployed as an Endpoint using Flask and a basic UI was used to expose the API for hitting and getting answers to queries.

Taking care of business and user experience we added a feature of "assisted video links" when and where required.

