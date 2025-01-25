# Batch-Image-Parallel-Processing-
C++ Parallel Image Processing Tool for a Dataset of Images. Applies multiple OpenCV functions on the whole batch using both Data and Task Parallelism for faster processing.

Commands for compiling this code:

If you have openCV4 already installed:
g++ -o ImageProcessing ImageProcessing.cpp -std=c++11 -I/usr/include/opencv4/ -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lpthread

if you don't have openCV installed initially:
g++ ImageProcessing.cpp -o ImageProcessing `pkg-config opencv4 --cflags --libs`
(tilde key `, not quote key '.
*/

We tested the program on 9000+ images Dataset and extracted great processing performance through Parallelism principles. Currently due to Github file upload restrictions /images folder in the repository contains 70-ish files while output folder creation is must locally (output folder can be empty as all input files will be uploaded there in result). You can download maximum folders from Kaggle Images Dataset etc and paste them in the images folder, it should make the program working with your desired number of files.

Sample Dataset Folder:
![WhatsApp Image 2025-01-25 at 11 06 11_576b2f90](https://github.com/user-attachments/assets/5993bbc4-015d-4c27-b6f6-6c165ff37786)
![WhatsApp Image 2025-01-25 at 11 06 36_7a837f21](https://github.com/user-attachments/assets/c4d8f8f8-be3b-4833-87d6-9e13e4d72f31)



Output Sample
Grayscale + Flip + Gaussian Blur + Edge Detection + Many More OPenCV operations!!
![image](https://github.com/user-attachments/assets/516851ae-da6a-4fbf-85f3-25356a3f3b7e)


