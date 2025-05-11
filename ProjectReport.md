DeepAI: Further Novel Applications of Deep Autoencoder Networks to Detect Anomalies in Large-Scale Financial and Accounting Data
Akhil Sreedhara, Andrew Aquino, Nicholas Gresh
New Jersey Institute of Technology, Newark, NJ
{as3638, ama347, ng584} @njit.edu
DS 677
11 May 2025
Video presentation link here*
GitHub Project Link
Abstract
In the modern world, financial and accounting data is vast and complex. With advances in technology, big data and its management is a large industry. With the scale of big data, systems are so large and complex that mistakes and fraud are inevitable. This has negative consequences for corporations, so there is a business incentive to account for, identify, manage, and prevent such risks. In a paper from (https://github.com/GitiHubi/deepAI) an Autoencoder deep learning model was used for accounting data analysis, a novel technique from the typical use of Autoencoder in image compression and reconstruction. In this paper, we implement and analyze the original model on the original dataset. This paper implements the analysis portion through examining the effects of tuning hyperparameters and model architecture, to identify what attributes make the model successful.
Introduction
First, we implemented the model from the original paper, then we built different models for the effects of different hyperparameters and model structure. In our work we tested different epochs, different learning rates, and a shallow model in comparison to the original deep multi-layered one.
Background on Financial and Accounting Processes
The model we created, like the original paper, was tested on accounting data. Such data is tracked by businesses as part of a system called ERP (enterprise resource planning), defined as “a software to help businesses streamline their core business processes” (https://www.sap.com/about/what-is-sap.html). ERP is used when a business has a need to track and edit complex sets of data. Such data included potentially thousands of data row entries, where each row contains a set number of column feature attributes. The types of features can be classified into the categories numeric and non-numeric, as defined below.
·       Numerical Features: number values, could be amount of money, weight, length, physical dimension, etc.  
·       Non-Numerical Features: features not represented in numerical format, could be name of company, time of day, location address, transaction type, cost center, etc.

Figure. From original paper, showing database structure being an accounting transaction
An example of an accounting process tracked in an ERP is shown in the figure. The transaction is illustrated in three levels: process, accounting, and database. The process level is simple: a supplier invoices the buyer $1000 for a good or service supplied, then the buyer sends the $1000 payment to the supplier, completing the transaction. The accounting level shows the tracking of the invoice as an expense, and tracks the invoice payment as credit, and when those equalize the transaction is complete. The database is where it starts to get complicated. Each transaction is a row, with columns tracking transaction data. This particular data tracks the company name, entry ID, fiscal year, type, data, and more.  
The figure shows an accounting transaction specifically, but an ERP is a broad term that can apply to a variety of businesses, including different kinds of procurements, orders, sales, and more. There are a variety of possible ERP implementations, but there is one common denominator highlighting the challenges of any ERP: successfully managing vast data sets with thousands of entries, and potentially hundreds of rows for the feature elements.
To the human eye, you might be able to catch a million-dollar outlier cost, but the challenge is when the outlier is hidden in the data. For example, consider data where each entry has dozens of features, and an outlier entry has a particular unusual combination of features values for that entry. The human mind isn’t equipped to observe that on our own, we need tools to do that. Machine Learning and particularly Deep Learning has a unique ability to find these hidden outliers.

Figure. SAP sales billing data
SAP is a prevalent ERP software used across various corporations today, SAP’s organization alone employs over 109,000+ employees across 157+ countries.  (https://www.sap.com/about/company.html#fast-factsl) In the figure is an example of a real life SAP business transaction. (https://community.sap.com/t5/enterprise-resource-planning-blog-posts-by-members/some-useful-tables-with-header-and-item-details/ba-p/13237198 ) A transaction in SAP is a command sent to the system, telling it to pull a specific data table according to specific criteria. This particular transaction is a command to pull a “sales billing table”. When SAP gets the command, it goes into the database and searches for each specific feature column needed for the transaction, as well as all the relevant row entries. It searches for sales billing data only, so it searches for entries that are entered when a sale is made. This data forms one large data table which is returned and shown on the screen. From here, the data can be filtered in each column to search for specific useful data points. The real-life usage of this could be to identify a particular sales entry within the larger returned data.
Integration of Autoencoder Model with Accounting Data
For the integration of our model with the accounting data, the first step was understanding the type of model we were seeking to build.

Figure. From OG paper, showing integration of accounting data with Autoencoder model
The Autoencoder model architecture is illustrated in the figure from the original paper. (cite figure). The dataset is a table with thousands of data entries and around 10 features, spread across two Excel spreadsheets. The non-numeric features must be encoded into numeric via one-hot encoding, so eventually all the data becomes numerical, and the data increases to hundreds of features. The data is then input into neural network layers, so it becomes “encoded” by being displayed in progressively fewer neurons. It eventually reaches its most compressed state of three neurons. Then, it is “decoded” symmetrically to the encoding. This means the number of layers and number of dimensions per layer are symmetrical, but each neuron doesn’t necessarily have the same weights and biases as its symmetric twin. However, the symmetric aspect of the Autoencoder allows data to be compressed smaller, then later expanded to replicate the input data. Any normal data should be output the same as before, but the outlier data output will change to become data that more closely resembles the normal data. At the end the reconstruction loss can be measured, where data with low loss is likely normal, whereas data with high loss are likely outliers. Thus, the loss metric is what can flag potential outliers, and further review can confirm the data to be true outliers.

Figure. From OG paper showing global vs. local anomaly
In the figure from the original paper (CITE HERE), the concept of “global” vs. “local” anomalies is illustrated. Each graph shows a plot of one feature vs. another feature for multiple data points.
A global anomaly could be like when transactions are usually $2,000 but now there is a $1,000,000 transaction. It refers to a data entry where one specific feature has a clear quantifiable outlier, visually observable on a graph for example.
A local anomaly is a data point where each entry feature is normal on its own, but the amalgamation of all the specific features together is unusual. Thus, the overall data point is an anomaly with everything put together. For example, if a raw wood materials supplier is invoicing an electronic component instead of the usual raw wood material. The raw wood supplier is in the system, electric components are in the system, but the fact that a wood supplier is buying electronics is an unusual combination.
In this paper, we implemented three different models to compare the reconstruction loss of the different model structures.  
Related Works
	One study that we came across that we thought would  be helpful is "Detection of Anomalies in Large-Scale Accounting Data using Deep Autoencoder Networks" by Zahra Zamanzadeh Darban. Their strategy is to use the reconstruction error generated by the autoencoder and further refine them with individual attribute probabilities of journal entries. This duel faceted scoring system increases the adaptability
Explanation of the model and what we did with it
The first step in creating the model is a Python notebook that brings in the initialization to allow all the necessary Python libraries to create a model. Then we load the data from a Github webpage where the Excel spreadsheet is stored. We plot the data table in Python for convenience so that the model is pulled correctly.
The actual data has some amount of normal data, but we also inject a small amount of global and local anomaly data (~0.03%) to see what the model detects for each anomaly type.
We plot the distribution of numerical values to see how the distribution of data points is numerically. Based on the plot, the DMBTR and WRBTR values are most always low in number, except for a few high number outliers.
 
Figure. one -hot encoding of non-numeric data
After implementing the data into the notebook, we have to perform data preprocessing on the data. For any non-numerical data, we turn that non-numeric feature into multiple one-hot features to represent all the unique string phrases of that non-numeric feature. The figure from the original paper has an example of this (CITE). Then, these new numerical features are combined with the original numerical features, thus resulting in a new dataset of all numerical features. The one-hot encoding transforms the data from around 10 features to 618 numerical features. 
We implemented an autoencoder network that compresses the data throughout multiple deep layers, thus it is a ‘deep learning’ model. The data starts out with the 618 features as input, then reduces neurons each layer into gradually smaller dimension multiples of 2. It reduces from 618 to 512, its output becoming a multiple of 2. Then it goes 512 to 256, to 128, 64, 32, 16, 8, 4. Finally, it goes from 4 to 3. At this point it reaches the center of the Autoencoder, the most compressed format. Then, it decodes symmetrically in reference to the the number of encoding layers, and how many neurons each layer has. Also for this model, key hyperparameters are established. A learning rate is established to affect how much each forward step modified the weights and biases of each neuron. Epoch number is established to determine how many passes the model has over the data. Mini-batches are established to improve efficiency instead of a pass over the whole date for each epoch.
The model trains in repeated passes, and we track the reconstruction loss over each epoch. We see that the model learns, and the loss goes down accordingly. Thus, it learns how to replicate the normal regular data and also convert outlier data into a normalized format. Now, in this context normalized means in terms of what made it an outlier before. The outliers are changed to be more normal in terms of a high-level combination of all the features in context. Throughout training, the reconstruction loss is tracked to measure how the model improves in detecting normal vs. outlier data. 
Experiments/Results
We test the original data on the original model and find that after training, outlier data can clearly be distinguished from normal data via measuring reconstruction loss. 

Figure. OG reconstruction loss

Figure. OG reconstruction loss for regular vs. the two outlier categories.
The measured loss by the end of training is quite low, ~.0003. Visually, we can see the reconstruction loss is close to zero for the regular data, but for the global and local outliers it is between 0.1 and 0.6. From this we have a clear visual indicator of regular vs. outlier, and we can set a horizontal boundary line to distinguish the two categories. 
We created a new model by changing the epoch hyperparameter from 5 to 50. 

Figure. Epoch 50 reconstruction error
We can see that past a certain point, the information gain from succeeding epochs decreases. We see that past 5 epochs, any knowledge gain is marginal, and at 8 epochs the model becomes more inaccurate for a couple epochs. 
We also created a new model by changing the learning rate hyperparameter from 0.001 to 0.01.

Figure. Learning rate .01 instead of .001 learning rate

We notice that the reconstruction error goes from a decimal previously, to now above one and reaching 25 at its peak. We can conclude keeping the learning rate initialized at that lower value yields the best return. 
Finally, we created a new model for shallow models instead of the deep learning multi-layered model.

In this model, we go directly from the 618 transformed numeric input features, directly to 3 neurons. We notice that over multiple epochs the error goes down, although the magnitude of error starts at .03 and approaches 0.016. This new model never reaches the original small error of ~0.003.
The technical innovation of this model is the ability to use an autoencoder to analyze financial and accounting data, instead of the usual other machine learning methods people have done before. In this model, we used the datasets provided in the original model to show that this novel approach could be used in real data pulled from SAP or another ERP platform for a businesses.
 
Potential future applications of this data could be used in tandem with other software, to directly interface with an ERP and set flagged entries up for automated or manual review. Perhaps a report of monthly bad entries could be sent to compliance. Or, perhaps an automated process could flag those for review first. There are both automatic and manual possibilities.
Limitations
There is the model limitations, then also the limitations of real-life data, and the complexity of data. 
The model itself may not be accurate enough to detect specific close data, so there might be false positives that cause delays via review of normal data. There could also be false negatives too, which run the risk of incorrect if not fraudulent data being missed and its consequences affecting a business. 
Then there is the limitation of real-life data. Perhaps real-life data isn’t accurate for the data points or is missing a data feature that if we did have, it could easily help identify outliers. 
Real life data is complex and although deep learning is abstract, so are real life principles. No model is perfect, each model is generally a specialized model to meet a certain thing, but again no model in science is perfect.
Ethics Statement
This is a model and no model is perfect, therefore applications of our different models should be used with discretion. One should also implement manual human analysis to check a few things. One to make sure it is ethical to even use the specific data you are planning to use. Two, if a conclusion is reached from the model analyzing data, following decisions should be made ethically. Third, in general this model shouldn’t be used for harm or evil.
Acknowledgements
We would like to thank the original model authors and paper authors. We would also like to thank NJIT and particularly professor Islam and professor Koutis for their support. 
References
https://github.com/GitiHubi/deepAI
https://community.sap.com/t5/enterprise-resource-planning-blog-posts-by-members/some-useful-tables-with-header-and-item-details/ba-p/13237198
https://www.sap.com/about/what-is-sap.html

Appendices
* inset all the figures here
 

 



