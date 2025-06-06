1.Write the addresses of the extracted multimodal features from the fourth part into the mlp. Ensure that the first column in the xlsx table is labeled as "label". 

If you want to remove other unnecessary information, you can modify X = dataz[dataz.columns[1:]] # Remove the labels 


2. After the data is filtered by the Lasso model, it is predicted and classified using the five-fold cross-validation method.