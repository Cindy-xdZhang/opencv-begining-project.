#include "Bitmap.h"
#include <stdio.h>
#include "mrf.h"
#include "MaxProdBP.h"
#include "ANN/ANN.h"
#include <iostream>

	int width,height;
	float* ImageF[3];//指针数组，指向
	float* MarkF[3];//指针数组，指向
    float* markupBack[3];//指针数组，指向
	float* markupFore[3];//指针数组，指向

	int lengthBack,lengthFore;
    int *labels; 

	ANNpointArray		dataPts_Fore;	// foreground data points  
	ANNpointArray		dataPts_Back;	// background data points  
	ANNpoint			queryPt;		// query point   不确定点
	ANNidxArray			nnIdx;			// near neighbor indices 临近点索引
	ANNdistArray		dists;			// near neighbor distances
	ANNkd_tree*			kdTree_Back;	// search structure Back ground points
	ANNkd_tree*			kdTree_Fore;    // search structure Fore ground points
	int			k_neighbors;			// number of nearest neighbors
    int			dim;					// dimension
    double		eps;					// error bound

//data cost function
MRF::CostVal dCost(int pix, int i){
   
   queryPt[0] = ImageF[0][pix];
   queryPt[1] = ImageF[1][pix];
   queryPt[2] = ImageF[2][pix];
   
   double value = 0.0;
   switch(i){
   
	   case 0:
		   kdTree_Back->annkSearch(						    // search
							queryPt,						// query point
							k_neighbors,					// number of near neighbors
							nnIdx,							// nearest neighbors (returned)
							dists,							// distance (returned)
							eps);							// error bound
		   value = dists[0];
		   break;

	   case 1:
		    kdTree_Fore->annkSearch(						    // search
							queryPt,						// query point
							k_neighbors,					// number of near neighbors
							nnIdx,							// nearest neighbors (returned)
							dists,							// distance (returned)
							eps);							// error bound
		   value = dists[0];
		   break;

	   default:
		   break;
   
   }
   return value;
}

int kk = 0;
//smooth cost function
MRF::CostVal fnCost(int pix1, int pix2, int i, int j){
   
   float R = ImageF[0][pix1] - ImageF[0][pix2];
   float G = ImageF[1][pix1] - ImageF[1][pix2];
   float B = ImageF[2][pix1] - ImageF[2][pix2];

  
   double C = R*R+G*G+B*B;

   C = sqrt(C);
   C = 1/(1+C);

   return abs(i-j)*C;

}


//////////////////////////////////////////////////////////////////////////////////
void main(int argc, char* argv[]){
   

   // argv[1] contains file name for input image
   // argv[2] contains file name for markup image
   // argv[3] contains file name for output image

	argv[1] = "monarch.bmp";
	argv[2] = "markup.bmp";
	argv[3] = "result.bmp";

   readBMP(argv[1], ImageF[0],ImageF[1],ImageF[2],width,height);
 
   readBMP(argv[2], MarkF[0],MarkF[1],MarkF[2],width,height);

   for(int k=0;k<3;k++){
   markupBack[k] = new float[width*height];
   markupFore[k] = new float[width*height];
   }
  
   //add user defined foreground and background 
   lengthBack = 0;
   lengthFore = 0;
   for(int i=0;i<width*height;i++){
	   //set forground points
       if(MarkF[0][i] == 1 && MarkF[1][i] == 0.0 && MarkF[2][i] == 0.0){
	     markupFore[0][lengthFore] = ImageF[0][i];
	   	 markupFore[1][lengthFore] = ImageF[1][i];
	   	 markupFore[2][lengthFore] = ImageF[2][i];
         lengthFore++;
	   }
	   //set background points
       if(MarkF[0][i] == 0 && MarkF[1][i] == 0 && MarkF[2][i] == 1){
	      markupBack[0][lengthBack] = ImageF[0][i];
	   	  markupBack[1][lengthBack] = ImageF[1][i];
	   	  markupBack[2][lengthBack] = ImageF[2][i];
  	      lengthBack++;
	   }
  }//end for
   



   dim=3;												  //(R G B) each
   k_neighbors=1;					                      // k nearest neighbor
   eps = 0.0;
   queryPt = annAllocPt(dim);					          // allocate query point,one point
   dataPts_Fore = annAllocPts(lengthFore, dim);		      // allocate data points
   dataPts_Back = annAllocPts(lengthBack, dim);		      // allocate data points 
   nnIdx = new ANNidx[k_neighbors];					      // allocate near neighbor indices
   dists = new ANNdist[k_neighbors];					  // allocate near neighbor dists
   labels = new int[width*height];
  


   //building the foreground KD tree
	for(int i=0;i<lengthFore;i++){
	   dataPts_Fore[i][0] = markupFore[0][i];
	   dataPts_Fore[i][1] = markupFore[1][i];
	   dataPts_Fore[i][2] = markupFore[2][i];
	}
    kdTree_Fore = new ANNkd_tree(dataPts_Fore,lengthFore,dim);


    //building the background KD tree
	for(int i=0;i<lengthBack;i++){
	   dataPts_Back[i][0] = markupBack[0][i];
	   dataPts_Back[i][1] = markupBack[1][i];
	   dataPts_Back[i][2] = markupBack[2][i];
	}
	kdTree_Back = new ANNkd_tree(dataPts_Back,lengthBack,dim);
  

	 //The standard use of the MRF  
	 MRF* mrf;
     EnergyFunction *energy;
	 DataCost *data = new DataCost(dCost);
	 SmoothnessCost *smooth = new SmoothnessCost(fnCost); 
	 energy = new EnergyFunction(data,smooth);
     float t;
	 mrf = new MaxProdBP(width,height,2,energy);
     mrf->initialize();  
     mrf->clearAnswer();
     mrf->optimize(3,t);  // run for 5 iterations, store time t it took 
     MRF::EnergyVal E_smooth = mrf->smoothnessEnergy();
     MRF::EnergyVal E_data   = mrf->dataEnergy();
    
	 for (int pix =0; pix < width*height; pix++ ) {
	 	 labels[pix]=mrf->getLabel(pix);//get the lables for each pixel
	 }			 
	 
	 for(int i=0;i<width*height;i++){
		 if(labels[i]==0 ){
		   ImageF[0][i] = 0;
		   ImageF[1][i] = 0;
		   ImageF[2][i] = 1;
		 }//set the background color based on labels 

		 if(MarkF[0][i] == 0 && MarkF[1][i] == 0 && MarkF[2][i] == 1){
		   ImageF[0][i] = 0;
		   ImageF[1][i] = 0;
		   ImageF[2][i] = 1;
		 }//set the background color based on user markup
	 }
   
   



   writeBMP(argv[3], width, height,ImageF[0],ImageF[1],ImageF[2] ); 

   /*
   for(int k=0;k<3;k++){
	    delete[] ImageF[k];
		delete[] MarkF[k];
	    delete[] markupBack[k];
		delete[] markupFore[k];
	}
    
	delete[] labels;
    delete[] nnIdx;
	delete[] dists;
	annDeallocPts(dataPts_Fore);
	annDeallocPts(dataPts_Back);
	annDeallocPt(queryPt);
	delete kdTree_Fore;
	delete kdTree_Back;
	annClose();
	delete mrf;
	*/

  return;
}