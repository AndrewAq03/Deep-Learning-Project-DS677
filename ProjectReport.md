**Objective: Create code, report, and video to explain our DL model.**

**Week 1 (due 4/20)**

- [x] Understand Milestone 3 requirements (milestone 3 due 5/11)

- [x] Understand the deepAI GitHub Page (includes paper, colab, and everything else) 

- [x] Understanding the objective and goal of deepAI (model architecture, parameters, inputs, outputs)

**Week 2 (due 4/27)**

- [ ] Replicate the papers results in our own code (implement code and replicate numerical results from model) 

- [ ] Discuss who is going to work on experimenting with each section (preprocessing) 

- [ ] Perhaps each person will experiment their own version of each section (one person does one type of preprocessing, one person another, etc.) 

- [ ] Each person experiment with code on their own end

- [ ] Only adding code to existing colab with consensus from group 

- [ ] Potentially meet with professor to clarify we are on the right page, and seek feedback

**Week 3 (due 5/4)**

- [ ] Writing the report document to explain our code and our model

- [ ] Drafting concept for the video

**Week 4 (due due 5/11, final project due)**

- [ ] Record video and finalize report

-----
DeepAI: Further Novel Applications of Deep Autoencoder Networks to Detect Anomalies in Large-Scale Financial and Accounting Data

Akhil Sreedhara, Andrew Aquino, Nicholas Gresh
NJIT, Newark, NJ [names] @ njit.edu
DS 677
11 May 2025
-----
Video presentation link here*
GitHub Project Link
----
Abstract 
In the modern world, financial and accounting data is vast and complex. With advances in technology, big data and its management is a large industry. With the scale of big data, systems are so large and complex that mistakes and fraud are inevitable. This has consequences for corporations, so there is a business incentive to account for, identify, and manage such risks. If possible, prevention is best. In a paper from (CITE original paper here) the researchers used an autoencoder for financial data and analysis, instead of the typical autoencoder usage for images. In this sense the original paper had a novel approach. In this paper, we seek to implement and analyze the model, specifically the effects of tuning hyperparameters and model architecture. We seek to build on the model and see if we can tweak it to learn what makes the model successful. 
Introduction 
First, we sought to implement the code from the original paper, then test our own tweaks and see the effect of different hyperparameters and model structures. Particularly, we tested different epochs, different learning rates, and a shallow model in comparison to the original deep multi-layered one. 
The model we created, like the original paper, was tested on financial and accounting data. Such data is tracked by corporations as part of a system called ERP (enterprise resource planning). (CITE HERE). SAP IS a big one (CITE HER). Such systems are vast data sets where each entry row has multiple columns as feature data points. ERP is used when a business has a need to track and edit complex sets of data. 
•	Numerical: Such features are often numerical to track money or encode organization. Could be the amount of money in currency, the length of a product, the weight of a product, or more 
•	Non-numerical: data could be the name of the person doing the entry, a time code, a location, a type of transaction, a cost center it is allocable to, or more. 
 
https://community.sap.com/t5/enterprise-resource-planning-blog-posts-by-members/some-useful-tables-with-header-and-item-details/ba-p/13237198
an example of what the data of a financial transaction would look like is the above. Here, a supplier invoices the buyer $1000 for a good or service supplied. Th buyer sends the $1000 payment. What this looks like in the data is that the specific important details of each transaction are tracked. Some important details include 
•	Full name of the supplier company
•	Specific internal code for that specific branch of the supplier company 
•	Time, date, or year of transaction
•	Where that invoiced amount will be sent to in terms of the buyer company. 
•	How much money was the invoice 
•	What currency was the invoice in
This is to name a few, but this is customizable to track most any business process. The challenge is regardless of business process, this creates large data sets with thousands of entries, and potentially hundreds of rows for the feature elements. To the human eye, you might be able to catch a multi-million dollar outlier cost, but the challenge is when the data is more hidden. Perhaps a data point will have dozens fo features, and an outlier has a particular unusual combination of features values for that point. The human mind isn’t equipped to observe that on our own, we need tools to do that. However, machine learning and particularly deep learning has a unique ability to find these local outliers. 
 
Here is a common example of what an SAP transaction will pull up in terms of a data table. This is a real-life example that someone’s job in a real company may use. A transaction in SAP is like a command sent to the system, telling it to pull a specific data table that tracks specific information. This particular transaction is a command to pull sales billing data. When SAP gets the command, it goes into the database and searches for each specific column needed for the transaction, as well as all the relevant row entries. It searches for sales billing data only, so it searches for entries that are entered when a sale is made. This data forms one large data table which is returned and shown on the screen. From here, the data can be filtered in each column to search for specific useful data points. The real life useage of this could be to identify a particular sales entry within the large data. 

First, we implemented the original code and ensured our understanding of it. First was understanding the type of model we were seeking to build. 
 
The model architecture is a typical autoencoder. Each data table set is a table with thousands of data entries, with dozens if not hundreds of features. Data that is characters must be reformatted into one hot, so eventually all the data becomes numerical. The data is fed into neural network layers, so it becomes “encoded” being displayed in gradually fewer nodes, in a compressed fashion. Then, it is “decoded” symmetrically to the encoding. Meaning the layers and number of dimensions are symmetrical, but each neuron doesn’t necessarily have the same weights and biases. But the part that is symmetrical allows it to be put into a compressed format, then unrolled. Any usual data should in theory roll out to be the same as before, and then the outlier when unrolled will change to become data points that look less like outliers. At the end the reconstruction loss can be measured, and anything with low loss is likely normal, whereas things with high loss are outliers. Thus, the loss is what can flag potential outliers, for real life review to confirm they are outliers. 
 
A global anomaly could be like when transactions are usually 2k but now there is a 1-million-dollar transaction. 
A local anomaly is a data point where each entry feature on its own is normal in the universe, but rather the amalgamation of all the specific features together is unusual. Thus, the overall data point is an anomaly with everything put together. For example, if a raw materials supplier is invoicing an electronic component instead of the usual raw material of wood. 
Second, we implemented three different other models to compare the seldom effects on reconstruction loss.  The analysis demonstrates the effects of hyperparameters and model architecture and results. 
Explanation of the model and what we did with it 
The first step in a model is a Python notebook that brings in the initialization to allow all the necessary Python libraries to create a model. We load the data from a webpage that goes to the site and downloads the Excel spreadsheet into the python notebook. We plot the data table in python for convenience so that the model is pulled correctly. 
The actual data has some amount of normal data, but we also inject a small amount of global and local anomaly data to see what the model detects for each anomaly type. 
We plot the distribution of numerical values to see how the distribution of data points is. Based on the plot, the dmbtr and wrbtr values are most always low in number, except for a few high number outliers. 

We have to do some data transformation fo the data, which is mostly the following. For any non-numerical data, we turn each feature into multiple one-hot features to represent all the unique string phrases of the original features. Then, these new numerical features are combined with the original numerical features, thus resulting in a new dataset of all numerical features. 


We implemented an autoencoder network that compresses the data throughout multiple layers, thus ‘deep’ learning. It starts out with the original 618 features, then reduces layers to neuron layers of multiple of 2. It reduces from 618 to 512. Then 512 to 256, 128, 64, 32, 16, 8, 4. Then it does 4 to 3. At this point it reaches the center of the autoencoder, the most compressed format. Then, it decodes symmetrically in terms of the number of decoding layers, and how many neurons each layer has. For this model, a learning rate is established to affect how much each step tweaks the weights and biases. Several epochs is established to determine how many passes it does over the data. Mini batches are established to improve efficiency instead of a pass over the WHOLE date for each epoch. 
The model trains repetitively, and we track the reconstruction loss over each epoch. We see that the model learns, and the loss goes down accordingly. Thus, it learns how to replicate the normal regular data and turn outlier data into a normalized format. Normalized in terms of, what made it an outlier before is changed to be more normal in a high-level combo of all the features context. 


 
For autoencoders, analysis needs to be on numerical data. In real life, this type of data might be in a word format, so we need a mathematical way to effectively turn the data into equivalent numerical format. Thus, comes in “one-hot encoding”. In this, instead of one feature of multiple words, it splits up into multiple features each for each unique word. For example, A/B/C one feature becomes feature a, feature b, feature c. When the datapoint is A the value is 1 else 0. When the data point is B is 1 and if not its 0. Thus, when the value is that specific string it assigns some custom weight to it, otherwise it gives it no weight. Another one hot method could be if true 1 else if not true 0. It depends on if you want to be penalized for not being something, or if you just don’t want that to have an effect . we create a class for the encoder and one for the decoder, and we see that each one has symmetrical number of layers, and each layer has a symmetrical number of neurons in each. What is Not symmetrical is that during the training, each specific neuron learns a unique weight and bias parameters. So those are NOT symmetrical across the autoencoder. But what it DOES do is enable the compression and expansion to keep the normal data as is and turn outlier data into a normalized version of that outlier. Later, we compare reconstruction loss, and the outlier are the ones that have the loss. We can set a numerical threshold to flag the data as outlier, if loss is above the threshold. This generally should identify outliers as is, and this is the step in real life where manual human review or a different code review would come in. This would identify outlier that are Local in addition to global, which is where it’s useful. 
Experiments/Results 
We test the normal data and find that it can identify 
We will create a new table for 50 epochs instead of 5
We have created a new model for 0.001 instead of .0001 learning rate 
We create a new model for shallow models instead of the deep learning deep layered model. 
Tech innovation is the ability to use an autoencoder to analyze financial and accounting data, instead of the usual other machine learning methods people have done before. In this model, we used the datasets provided in the original, real accounting data to show that this novel approach could be used in real data pulled from SAP or another ERP platform for a businesses

Potential future applications of this data could be this used in tandem with another software, to directly interface with an ERP and set flagged entries up for automated or manual review. Perhaps a report of monthly bad entries could be sent to compliance. Or, perhaps an automated process could flag those for review first. There are both automatic and manual possibilities. 
Limitations
There is the model limitations, then also the limitations of real-life data, and the complexity of data 
The model itself may not be accurate enough to detect specific close data, so there might be false positive that cause delays. It could be false negatives too which slip through the cracks and are more severe
Then there is limitation of real-life data. Perhaps real-life data isn’t accurate for the data points or is missing a data feature that if we did have, could easily help identify outliers. 
Real life data is complex and although deep learning is abstract, so are real life principles. No model is perfect, each model is generally a specialized model to meet a certain thing, but again no model in science is perfect. 
Ethics Statement 
This is a model and no model is perfect, therefore applications of our different models should be used with discretion. One should also implement manual human analysis to check a few things. One to make sure it is ethical to even use the specific data you are planning to use. Two, if a conclusion is reached from the model analyzing data, according following decisions should be made ethically. Third, in general this model shouldn’t be used for harm or evil. 
Acknowledgements 
f
References 
https://github.com/GitiHubi/deepAI
https://community.sap.com/t5/enterprise-resource-planning-blog-posts-by-members/some-useful-tables-with-header-and-item-details/ba-p/13237198
https://www.sap.com/about/what-is-sap.html
Appendices 
f

----


