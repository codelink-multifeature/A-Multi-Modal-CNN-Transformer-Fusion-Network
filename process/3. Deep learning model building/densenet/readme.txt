1.The data template can be found in the folder.

 
2. First, run the 'data_acq.py' script to perform the preprocessing of 3D data. 

If the substring 'A' is found at positions 3 to 4: # Check if it is of type A  

You need to modify the data yourself to ensure that the output characters represent the category.  
Then, assign 'a' as k[5:12].  

You also need to modify the data yourself to ensure that the output is the patient number. 

3. Train the Densenet model, obtain the weight file, and place the weight file in the same folder. 

4. Import the weight address into the feature extraction of Densenet, obtain the output template xlsx file, and classify the output template into categories A and B based on the patient number, resulting in two templates.

