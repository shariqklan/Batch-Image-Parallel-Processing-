/*Project Members:

 22K-4285 (Muhammad Shariq Nadeem) (Team Lead)
 22K-4446 (Saad Arfan)
 22K-4283 (Muhammad Taha Imran)
 
 Section: BCS-4B
 
Commands for compiling this code:

If you have openCV4 already installed:
g++ -o ImageProcessing ImageProcessing.cpp -std=c++11 -I/usr/include/opencv4/ -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lpthread

if you don't have openCV installed initially:
g++ ImageProcessing.cpp -o ImageProcessing `pkg-config opencv4 --cflags --libs`
(tilde key `, not quote key '.
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cctype>
#include <pthread.h>
#include <semaphore.h>
#include <thread>
#include <unistd.h>

using namespace std;
using namespace cv;

const int MAX_IMAGES = 3000;

// Rubric Point 5: declaring mutex and semaphores

/*one binary semaphore for each image to ensure only one operation is being performed on an image at any single time. Has to be global as it is accessed by multiple functions*/

vector<sem_t> guardians;
pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

int operationSpecify[2];//index 0 has rotate angle, 1 has flipCode

/*Rubric Point 6: Multiple shared Data Structure, an buffer type array of images that is shared and accessed concurrently throughout program by multiple threads.*/

Mat images[MAX_IMAGES];

int numLoadedImages = 0;  //count for successful loaded images

/*Rubric Point 2: Input data generation from a folder of images to the array of images using multithreading*/

void* loadImage(void* arg)//adds a single image to the images Mat array, uses mutex lock when adding the image.
{
	string* filePath = (string*)arg;
	Mat img = imread(*filePath);
	if(!img.empty())
	{
		/*Rubric Point 6: Critical Section dealing by synchronization*/
		
		pthread_mutex_lock(&mtx);
		if (numLoadedImages < MAX_IMAGES) {
                   images[numLoadedImages++] = img;}
		pthread_mutex_unlock(&mtx);
		cout<<"Loaded image: "<<*filePath<<endl;
	}
	else
	{
		/*Rubric Point 9: Error checking/debugging*/
		
		cerr<<"Failed to load image: "<<*filePath<<endl;
	}
	pthread_exit(NULL);
}

//Welcome Window Interface

void displayMenu()//displays the menu and options for the user
{
	cout<<"\n\nWelcome to Image Processing Program!"<<endl;
	cout<<"Select the operations you want to perform: "<<endl;
	cout<<"g: Grayscale\n";
	cout<<"f: Flip\n";
	cout<<"r: Rotate\n";
	cout<<"hc: High Contrast | lc: Low Contrast\n";
	cout<<"hb: High Brightness | lb: low Brightness\n";//all other factors can also be influenced somewhat by user input
	cout<<"gb: Gaussian Blur | br: Background Remover | ed: Edge Detector\n\n";//bg remover can be made more specific (user input)

}

//Function to add options selected by user in a vector (actions to be performed on image set

vector<string> getUserOperations(int numOptions)//obtains and returns string vector of all operations based on the number specified by user
{
	vector<string> selectedOperations;
	int index=0;
	while(selectedOperations.size() < numOptions)//until all operations aren't obtained from user
	{
		//Label used
		again:
		
		string option;
		cout<<"Enter option "<<selectedOperations.size()+1<<": ";
		cin>>option;
		
		/*Rubric Point 8: Use of stl or built-in data structures (transform() )*/
		transform(option.begin(), option.end(), option.begin(), ::tolower);
		if(option=="g" || option== "f" || option=="r" || option=="hc" || option == "br" || option == "gb" || option == "ed" || option=="hb"||option=="lc" ||option=="lb")
		{
		
			if(index>0)
			{
				/*Rubric Point 9: Error checking/debugging*/
				for(auto iter=0; iter< index; ++iter)//for loop to ensure no duplicate operations entered goes it again if yes
				{
					if(option==selectedOperations[iter])
					{
						cout<<"\n\nOption already selected. Please try another one."<<endl;
						//Jump to a label
						goto again;
					}
				}
				
			}
			if(option == "f"){
				int reqFlip;
				cout << "Enter flip code (0 for vertical, 1 for horizontal, -1 for both): ";
				cin >> reqFlip;
				operationSpecify[1] = reqFlip;
			}
			if(option == "r"){
				int reqAngle;
				cout << "Enter angle to rotate all your images by (Anticlockwise): ";
				cin >> reqAngle;
				operationSpecify[0] = reqAngle;
			}
			selectedOperations.push_back(option); //if not duplicate, adds operation to list of operations.
			index++;
		}
		
		/*Rubric Point 9: Error checking/debugging*/
		else{   //if invalid input entered
			cout<<"Invalid option! Please select from g,f,r,hc,lc,hb,or lb"<<endl;
		}
	}
	return selectedOperations;//returns array of selected operations by user
}
			
//Initialization of 8 different functions that could be performed on the images set

//Rotate function
void imgRotater(int angle, int i){//rotates a single image according to specified angle

			//first get image width and height
			int height = images[i].rows;
			int width = images[i].cols;

			//then, find the center of the image to rotate about
			Point2f image_center(width / 2.f, height / 2.f);

			//make a copy to rotate on:
			Mat rotation_mat = getRotationMatrix2D(image_center, angle, 1.0);

			//now we find the absolute values of cosine and sine from our rotation matrix
			double abs_cos = abs(rotation_mat.at<double>(0,0));
			double abs_sin = abs(rotation_mat.at<double>(0,1)); 			

			//now to find the new width and height of the final image:
			int new_width = int(height * abs_sin + width * abs_cos); 
			int new_height = int(height * abs_cos + width * abs_sin);

			//now we adjust the rotation matrix for the new image center
			rotation_mat.at<double>(0,2) += new_width / 2.0 - image_center.x;
			rotation_mat.at<double>(1,2) += new_height / 2.0 - image_center.y;

			//lastly we rotate the image with the calculated new bounds and translated 
			//rotation matrix
			Mat rotated_mat;
			warpAffine(images[i], rotated_mat, rotation_mat, Size(new_width, new_height));
	
			images[i] = rotated_mat;	
}

//Flip image
void imgFlipper(int flipCode, int i){

		flip(images[i], images[i], flipCode);
	
}

//Grayscale image
void imgGrayer(int i){

		cvtColor(images[i],images[i],COLOR_BGR2GRAY);
	
}
	
//High contrast image
void histogramContraster(int i){// a more intelligent contrast increase giiving a more balanced look

		cvtColor(images[i], images[i], COLOR_BGR2YCrCb); //this is done to isolate the intensity channel
		//(Y) so we can manipulate it independently while keeping color untouched

		vector<Mat> channels;//now we split the image to get the Y channel
		split(images[i], channels);

		//we equalize the histogram of only the Y channel:
		equalizeHist(channels[0], channels[0]);

		//we merge the 3 channels back to YCrCb color space
		merge(channels, images[i]);

		//lastly, we convert the image back to BGR color space to allow it to be displayed normally
		cvtColor(images[i], images[i], COLOR_YCrCb2BGR);
}

//Low contrast image
void lowContraster(int i){

	images[i].convertTo(images[i], -1, 0.5, 0);//can add intesity of contrast later if needed
	
}

//High and Low Brightness image
void brightnesser(int i, string req){
	if(req == "lb"){
	
			images[i].convertTo(images[i], -1, 1, -100);
		
	}
	else{ //high brightness

			images[i].convertTo(images[i], -1, 1, 100);
		
	}
}

//Edge Detection image
void edger(int i){

	Mat resultCanny;
	Canny(images[i],resultCanny,80, 240);
	images[i] = resultCanny;
	
	
}

//Gaussian Blur image
void imgGaussBlur(int i){

	int kernel = 5;//size of gaussian kernel (neighborhood for blurring)
	int sigMax = 13;//standard deviation of gaussian distribution used for blurring
	GaussianBlur(images[i],images[i],Size(kernel, kernel),sigMax);
	
}

//Background Remove image
void imgBGRemover(int i){//uses color thresholding

		Mat hsvImage;
		cvtColor(images[i],hsvImage, COLOR_BGR2HSV);//the hsv color scheme works better for color segmentation
		
		Scalar lowerBound(0,100,100); //these values are hue, saturation, and HSV value
		Scalar upperBound(10,255,255);

		//now we create a mask to isolate the foreground
		Mat mask;
		inRange(hsvImage, lowerBound, upperBound, mask);

		Mat foreground;
		images[i].copyTo(foreground, mask);

		images[i] = foreground;
}

//function for partitioning images into chunks and assign it to consumer threads

int getOptimalChunkSize(int numImages){

	int optimal, minimum = 2;
	int coreCount = std::thread::hardware_concurrency();
	optimal = max(minimum, (numImages / coreCount));

	if(numImages < (coreCount*10)){//smaller datasets, smaller chunk size for smaller thread overhead
		return (optimal / 2);
	}
	else if(coreCount <= 4){//low number of cores: use ideal size as is
		return optimal;
	}
	else{//for larger datasets and more cores, use a higher chunk size for better resource utilization
		return (optimal * 2);
	}

}

//Function which is accessed by child_thread (local to each consumer thread), this function calls the desired user operation to be applied on the image

void handleOperations(vector<string>& reqOperations, int image_index, int op_index){ 
//takes the array 
//of operations to be performed, and performs the specified index's operation (just one) on a specific image.
	string task = reqOperations[op_index];
	cout<<"\nImage index: "<<image_index<<"\nOperation: "<<task<<endl;
		sem_wait(&guardians[image_index]); //this ensures any other operation on this image will have to wait
		//Rubric Point 9: Testing evidence/ debugging
			cout << "Performing operation #" <<op_index << " on image #" << image_index<< endl;
			if(task == "r"){
				imgRotater(operationSpecify[0], image_index);
			}
			else if(task == "f"){
				imgFlipper(operationSpecify[1], image_index);
			}
			else if(task == "g"){
				imgGrayer(image_index);
			}
			else if(task == "hc"){
				histogramContraster(image_index);
			}
			else if(task == "lc"){
				lowContraster(image_index);
			}
			else if(task == "lb" || task == "hb"){
				brightnesser(image_index, task);
			}
			else if(task == "ed"){//edge detection
				edger(image_index);
			}
			else if(task == "br"){//background remove
				imgBGRemover(image_index);
			}
			else if(task == "gb"){//gaussian blur
				imgGaussBlur(image_index);
			}
			cout << "Completed operation #" <<op_index << " on image #" << image_index<< endl;
		sem_post(&guardians[image_index]);
}


/*Rubric point 4: Data and Task Parallelism. 
Data Parallelism: Each consumer thread will call this runner function on its set of chunk parallel (Chunk of data calling same task).
Task Parallelism: Each thread in runner function will further create some thread_local child_threads which would be the size of the chunk. This chunk will be calling different operations parallel on the same chunk images, abiding the synchronization conditions within the chunk using semaphores*/


void runner(int id, vector<string> Op,int num, int startIndex, int endIndex){

	int size = (endIndex - startIndex);//get number of images to work on
	
	//Usage of thread_local
	//creates and initializes a 2D array, and variable of bool local to each thread.
	thread_local bool** progress;
	thread_local bool done= false;
	progress = new bool*[size];//progress is an array of n bool arrays of n is number of images

    for (int i = 0; i < size; ++i) {//for each image, a bool array in progress is made, and isBusy for
	//that image's index is made, all initialized to false

        progress[i] = new bool[num];
        // Initialize each element to false
        for (int j = 0; j < num; ++j) {
            progress[i][j] = false;
        }
    }

	//Rubric 9: Debugging and Error handling
	cout<<"\nProgress & isBusy initialized\n\n";
	thread_local vector<thread> childs;// each parent thread in main that called runner gets a vector of children threads
	cout<<"\nchildren threads created\n";
	
	for(int i = startIndex; i < endIndex; i++){//initializes all sempahores as binary semaphores
		if(sem_init(&guardians[i], 0, 1) == -1){
			perror("semaphore initizalization failure");
			exit(EXIT_FAILURE);
		} 
	}

	while(done==false){

		for(int i=0; i<size; i++){
			for(int j=0; j<num; j++){
				if(progress[i][j]!=false)
					done=true;
				else{
					done=false;
					break;
				}
			}
		}
		cout<<"\nValue of done: "<<done<<endl;
	
		if(done==true)
			break;
	// __________________________________________________________________________________________________
    	

		for(int i=0; i<size; i++){
			for(int j=0; j<num; j++){
				if(progress[i][j]){// if operation j has already been successfully performed on image i
				//, don't repeat it
					continue;
				}
					
				childs.emplace_back(handleOperations,ref(Op),i+startIndex,j);
				cout<<"\nChild thread: "<<i<<"successfully executed operation: "<<Op[j]<<endl;
				progress[i][j]=true;
			}
		}

	}

	for(int i = 0; i < size; i++){//destroy all semaphores after image editing is complete
			sem_destroy(&guardians[i]);	
		}
	// __________________________________________________________________________________________________	
	
	cout<<"\nAll child threads done Task Parallelism!\n\n";
	
	//join all threads
	for (auto& thread : childs) {
    	thread.join();
    }

}	

int main(){
	
	displayMenu();
	int num;
	enter:
	cout<<"Enter the number of options you want to perform: ";
	cin>>num;
	//Rubric 9: Error handling
	if(num<0 || num>10)
	{
		cout<<"Please enter correct number of options\n";
		goto enter;
	}

	else if(num!=0)
	{
		vector<string> selectedOperations= getUserOperations(num);
		cout<<"Selected operations: "<<endl;
		for(string op : selectedOperations)//print all the operations selected by user
		{
			cout<< op<<" ";
		}
		cout<<endl;

		string folderPath= "images/";
		pthread_t inputThreads[MAX_IMAGES];
		
		vector<string> fileList;
		glob(folderPath, fileList);
		int numImages = min((int)fileList.size(), MAX_IMAGES);//obtains the number of files and their names
		
		//Rubric 5: Multiple producer threads in images array
		for (int i = 0; i < numImages; ++i) {//loads images using multi-threading
        string* filePath = new string(fileList[i]);
        	pthread_create(&inputThreads[i], NULL, loadImage, (void*)filePath); //use threads to get images
		}

		for (int i = 0; i < numImages; ++i) {
			pthread_join(inputThreads[i], NULL);
		}

		int num_thread = getOptimalChunkSize(numImages);
		int images_per_thread = numImages/num_thread;

		vector<thread> consumerThreads;
		int startIndex = 0;
		int endIndex= images_per_thread;
		
		guardians.resize(numImages);//make the vector of semaphores have one semaphore per image

		//Rubric 5: Multiple consumer threads to consumer image chunk and apply user selected operations
		for(int i=0; i<num_thread ; ++i){//for each thread, 
			cout<<startIndex<<"->"<<endIndex<<endl;
			if(i==(num_thread-1)){//once last thread reached, the ending index is the number of images, this ensures
			//if any extra remainder images left, they're included too, eg 10 images, last thread gets 6,7,8, and 9
			//*(4 images, not 3)
				endIndex = numImages;
			}

			//at the ith position of consumerThreads, creates a thread that calls the runner function with
			//arguments: int id, vector<string> Op,int num, int startIndex, int endIndex
			consumerThreads.emplace_back(runner,i,ref(selectedOperations),num, startIndex, endIndex);
				
			startIndex =endIndex;//now moves to the next set of 3 images
			endIndex += images_per_thread;
		}
		
		  for (auto& thread : consumerThreads) {//join all threads now
        		thread.join();
    		}


		//Rubric 3: Real world output generation of operations applied images in a separate folder 'output'
		string outputPath="output/";
		for(int i=0; i< numImages; i++)
		{
			string outputFilePath= outputPath + "image_" + to_string(i) + ".jpg";
			imwrite(outputFilePath,images[i]);
			cout<<"Saved image: "<<outputFilePath<<endl;
		}
		cout << "Number of images loaded: " << numLoadedImages << endl;
	}
	cout<<"\n\nThank You for choosing our application!\n";
	return 0;
}
