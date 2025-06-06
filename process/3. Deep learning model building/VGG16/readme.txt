1.The data template can be found in the folder.
 
2. First, run data_acq.py to perform the preprocessing of 2D slice data. If i[3:4] is 'A': # Check if it is of type A. You need to modify the data yourself to ensure that the output characters represent the category. a = k[5:12] You also need to modify the data yourself to ensure that the output is the patient number. 


3. Train the Vgg16 model, obtain the weight file, and place the weight file in the same folder. 


4. Import the weight address into the feature extraction process of vgg16, obtain the output template xlsx file, and classify the output template into categories A and B based on the patient number, resulting in two templates.

