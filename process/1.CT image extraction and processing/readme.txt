1. "data" is used to record the address of the original CT data.

"mask" is used to record the address of the CT label. 


2. The 2Dslice is used to process slices into 2D. The input is the addresses of data and mask, and the final output is the ROI area with a size of 32*32*3. 


3. 3D volume is used to process slices into 3D. The input is the addresses of data and mask, and the final output is an ROI region of size 32*32*32.