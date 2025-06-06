1.Enter the address of the XLSX file containing the patient information collected by the doctor. 

The template is inside the folder, in second_A_group and second_B_group. 


2. Input the address of the CT images. Ensure that within one subfolder, there is data and mask information for one patient. See Input data stores styles. 


3. The range in 
`patient_number = patient_dir.split('\\')[-1][:7]` 
needs to be adjusted to ensure that the output is the patient number collected by the doctor.

 
4. The output templates and data storage styles are placed inside the folder.