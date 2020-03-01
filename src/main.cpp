
#include "header.hpp"
#include <ctime>
#include <string>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace viennacl::linalg;



void log( const std::string &text )
{
	//std::string fileName = "log_file_"+boost::lexical_cast<std::string>(::getpid())+".txt";
	std::ofstream log_file(
			"log_file.txt", std::ios_base::out | std::ios_base::app);
    log_file << text << std::endl;
}

void log_cost( const std::string &text )
{
	//std::string fileName = "log_file_"+boost::lexical_cast<std::string>(::getpid())+".txt";
	std::ofstream log_file(
			"log_cost.txt", std::ios_base::out | std::ios_base::app);
    log_file << text << std::endl;
}


Eigen::SparseMatrix<float, Eigen::RowMajor, int> Dmatrix(cv::Mat& Src,
		cv::Mat & Dest, int rfactor) {
	int dim_srcvec = Src.rows * Src.cols;
	int dim_dstvec = Dest.rows * Dest.cols;

	Eigen::SparseMatrix<float, Eigen::RowMajor, int> _Dmatrix(dim_srcvec,
			dim_dstvec);
	for (int i = 0; i < Src.rows; i++) {
		for (int j = 0; j < Src.cols; j++) {
			int LRindex = i * Src.cols + j;
			for (int m = rfactor * i; m < (i + 1) * rfactor; m++) {
				for (int n = rfactor * j; n < (j + 1) * rfactor; n++) {
					int HRindex = m * Dest.cols + n;
					_Dmatrix.coeffRef(LRindex, HRindex) = 1.0 / rfactor
							/ rfactor;
					//std::cout<<"_Dmatrix.coeffRef(LRindex,HRindex) = "<<1.0/rfactor/rfactor<<", rfactor = "<<rfactor<<std::endl;
				}
			}
		}
	}

	return _Dmatrix;
}

Eigen::SparseMatrix<float, Eigen::RowMajor, int> Hmatrix(cv::Mat & Dest,
		const cv::Mat& kernel) {

	int dim_dstvec = Dest.rows * Dest.cols;

	Eigen::SparseMatrix<float, Eigen::RowMajor, int> _Hmatrix(dim_dstvec,
			dim_dstvec);

	for (int i = 0; i < Dest.rows; i++) {
		for (int j = 0; j < Dest.cols; j++) {
			int index = i * Dest.cols + j;

			int UL = (i - 1) * Dest.cols + (j - 1);
			if (i - 1 >= 0 && j - 1 >= 0 && UL < dim_dstvec)
				_Hmatrix.coeffRef(index, UL) = kernel.at<float>(0, 0);
			int UM = (i - 1) * Dest.cols + j;
			if (i - 1 >= 0 && UM < dim_dstvec)
				_Hmatrix.coeffRef(index, UM) = kernel.at<float>(0, 1);
			int UR = (i - 1) * Dest.cols + (j + 1);
			if (i - 1 >= 0 && j + 1 < Dest.cols && UR < dim_dstvec)
				_Hmatrix.coeffRef(index, UR) = kernel.at<float>(0, 2);
			int ML = i * Dest.cols + (j - 1);
			if (j - 1 >= 0 && ML < dim_dstvec)
				_Hmatrix.coeffRef(index, ML) = kernel.at<float>(1, 0);
			int MR = i * Dest.cols + (j + 1);
			if (j + 1 < Dest.cols && MR < dim_dstvec)
				_Hmatrix.coeffRef(index, MR) = kernel.at<float>(1, 2);
			int BL = (i + 1) * Dest.cols + (j - 1);
			if (j - 1 >= 0 && i + 1 < Dest.rows && BL < dim_dstvec)
				_Hmatrix.coeffRef(index, BL) = kernel.at<float>(2, 0);
			int BM = (i + 1) * Dest.cols + j;
			if (i + 1 < Dest.rows && BM < dim_dstvec)
				_Hmatrix.coeffRef(index, BM) = kernel.at<float>(2, 1);
			int BR = (i + 1) * Dest.cols + (j + 1);
			if (i + 1 < Dest.rows && j + 1 < Dest.cols && BR < dim_dstvec)
				_Hmatrix.coeffRef(index, BR) = kernel.at<float>(2, 2);

			_Hmatrix.coeffRef(index, index) = kernel.at<float>(1, 1);
		}
	}

	return _Hmatrix;
}

Eigen::SparseMatrix<float, Eigen::RowMajor, int> Mmatrix(cv::Mat &Dest,
		float deltaX, float deltaY) {
	int dim_dstvec = Dest.rows * Dest.cols;

	Eigen::SparseMatrix<float, Eigen::RowMajor, int> _Mmatrix(dim_dstvec,
			dim_dstvec);

	for (int i = 0; i < Dest.rows; i++) {
		for (int j = 0; j < Dest.cols; j++) {
			if (i < (Dest.rows - std::floor(deltaY))
					&& j < (Dest.cols - std::floor(deltaX))
					&& (i + std::floor(deltaY) >= 0)
					&& (j + std::floor(deltaX) >= 0)) {
				int index = i * Dest.cols + j;
				int neighborUL = (i + std::floor(deltaY)) * Dest.cols
						+ (j + std::floor(deltaX));
				int neighborUR = (i + std::floor(deltaY)) * Dest.cols
						+ (j + std::floor(deltaX) + 1);
				int neighborBR = (i + std::floor(deltaY) + 1) * Dest.cols
						+ (j + std::floor(deltaX) + 1);
				int neighborBL = (i + std::floor(deltaY) + 1) * Dest.cols
						+ (j + std::floor(deltaX));

				if (neighborUL >= 0 && neighborUL < dim_dstvec)
					_Mmatrix.coeffRef(index, neighborUL) = (i
							+ std::floor(deltaY) + 1 - (i + deltaY))
							* (j + std::floor(deltaX) + 1 - (j + deltaX));
				if (neighborUR >= 0 && neighborUR < dim_dstvec)
					_Mmatrix.coeffRef(index, neighborUR) = (i
							+ std::floor(deltaY) + 1 - (i + deltaY))
							* (j + deltaX - (j + std::floor(deltaX)));
				if (neighborBR >= 0 && neighborBR < dim_dstvec)
					_Mmatrix.coeffRef(index, neighborBR) = (i + deltaY
							- (i + std::floor(deltaY)))
							* (j + deltaX - (j + std::floor(deltaX)));
				if (neighborBL >= 0 && neighborBL < dim_dstvec)
					_Mmatrix.coeffRef(index, neighborBL) = (i + deltaY
							- (i + std::floor(deltaY)))
							* (j + std::floor(deltaX) + 1 - (j + deltaX));
			}

		}
	}

	return _Mmatrix;
}

void motionMat(std::vector<Mat>& motionVec, size_t image_count, size_t rfactor,
		bool clockwise) {

	size_t quotient, remainder;

	if (clockwise) {
		for (size_t i = 0; i < image_count; i++) {
			Mat motionvec = Mat::zeros(3, 3, CV_32F);
			motionvec.at<float>(0, 0) = 1;
			motionvec.at<float>(0, 1) = 0;
			motionvec.at<float>(1, 0) = 0;
			motionvec.at<float>(1, 1) = 1;
			motionvec.at<float>(2, 0) = 0;
			motionvec.at<float>(2, 1) = 0;
			motionvec.at<float>(2, 2) = 1;

			quotient = floor(i / 1.0 / rfactor);
			remainder = i % rfactor;

			if (quotient % 2 == 0)
				motionvec.at<float>(0, 2) = remainder / 1.0 / rfactor;

			else
				motionvec.at<float>(0, 2) = (rfactor - remainder - 1) / 1.0
						/ rfactor;

			motionvec.at<float>(1, 2) = quotient / 1.0 / rfactor;

			motionVec.push_back(motionvec);

//			std::cout << "image i = " << i << ", x motion = "
//					<< motionvec.at<float>(0, 2) << ", y motion = "
//					<< motionvec.at<float>(1, 2) << std::endl;
		}
	} else {
		for (size_t i = 0; i < image_count; i++) {
			Mat motionvec = Mat::zeros(3, 3, CV_32F);
			motionvec.at<float>(0, 0) = 1;
			motionvec.at<float>(0, 1) = 0;
			motionvec.at<float>(1, 0) = 0;
			motionvec.at<float>(1, 1) = 1;
			motionvec.at<float>(2, 0) = 0;
			motionvec.at<float>(2, 1) = 0;
			motionvec.at<float>(2, 2) = 1;

			quotient = floor(i / 1.0 / rfactor);
			remainder = i % rfactor;
			if (quotient % 2 == 0)
				motionvec.at<float>(1, 2) = remainder / 1.0 / rfactor;

			else
				motionvec.at<float>(1, 2) = (rfactor - remainder - 1) / 1.0
						/ rfactor;

			motionvec.at<float>(0, 2) = quotient / 1.0 / rfactor;

			motionVec.push_back(motionvec);

		}
	}

}

Eigen::SparseMatrix<float, Eigen::RowMajor, int> sparseMatSq(
		Eigen::SparseMatrix<float, Eigen::RowMajor, int>& src) {

	Eigen::SparseMatrix<float, Eigen::RowMajor, int> A2(src.rows(), src.cols());

	for (int k = 0; k < src.outerSize(); ++k) {
		for (typename Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator innerit(
				src, k); innerit; ++innerit) {
			//A2.insert(innerit.row(), innerit.col()) = innerit.value() * innerit.value();
			A2.insert(k, innerit.index()) = innerit.value() * innerit.value();
			//A2.insert(innerit.row(), innerit.col()) = 0;
		}
	}
	A2.makeCompressed();
	return A2;

}

Eigen::SparseMatrix<float, Eigen::RowMajor, int> ComposeSystemMatrix(
		cv::Mat& Src, cv::Mat& Dest, const cv::Point2f delta, int rfactor,
		const cv::Mat& kernel,
		Eigen::SparseMatrix<float, Eigen::RowMajor, int>& DMatrix,
		Eigen::SparseMatrix<float, Eigen::RowMajor, int> &HMatrix,
		Eigen::SparseMatrix<float, Eigen::RowMajor, int> &MMatrix) {

	int dim_srcvec = Src.rows * Src.cols;
	int dim_dstvec = Dest.rows * Dest.cols;

	//float maxPsfRadius = 3 * rfactor * psfWidth;

	Eigen::SparseMatrix<float, Eigen::RowMajor, int> _DHF(dim_srcvec,
			dim_dstvec);

	DMatrix = Dmatrix(Src, Dest, rfactor);
	HMatrix = Hmatrix(Dest, kernel);
	MMatrix = Mmatrix(Dest, delta.x, delta.y);

	_DHF = DMatrix * (HMatrix * MMatrix);

	_DHF.makeCompressed();

	return _DHF;
}

void Normalization(Eigen::SparseMatrix<float, Eigen::RowMajor, int>& src,
		Eigen::SparseMatrix<float, Eigen::RowMajor, int>& dst) {
	for (Eigen::Index c = 0; c < src.rows(); ++c) {
		float colsum = 0.0;
		for (typename Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator itL(
				src, c); itL; ++itL)
			colsum += itL.value();

		for (typename Eigen::SparseMatrix<float, Eigen::RowMajor, int>::InnerIterator itl(
				src, c); itl; ++itl)
			dst.coeffRef(itl.row(), itl.col()) = src.coeffRef(itl.row(),
					itl.col()) / colsum;
	}
}

void Gaussiankernel(cv::Mat& dst) {
	int klim = int((dst.rows - 1) / 2);

	for (int i = -klim; i <= klim; i++) {
		for (int j = -klim; j <= klim; j++) {
			float dist = i * i + j * j;
			dst.at<float>(i + klim, j + klim) = 1 / (2 * M_PI) * exp(-dist / 2);
		}
	}

	float normF = cv::sum(dst)[0];
	dst = dst / normF;
}

void GenerateAT(cv::Mat& Src, cv::Mat& Dest, int imgindex,
		std::vector<Mat>& motionVec, cv::Mat &kernel, size_t rfactor,
		Eigen::SparseMatrix<float, Eigen::RowMajor, int>& DMatrix,
		Eigen::SparseMatrix<float, Eigen::RowMajor, int> &HMatrix,
		Eigen::SparseMatrix<float, Eigen::RowMajor, int> &MMatrix,
		std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor, int> >& A,
		std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor, int> >& AT,
		std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor, int> >& B,
		std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor, int> >& BT,
		std::vector<viennacl::compressed_matrix<float> >& DHF,
		std::vector<viennacl::compressed_matrix<float> >& DHFT,
		std::vector<viennacl::compressed_matrix<float> > &DHF2,
		std::vector<viennacl::compressed_matrix<float> > &DHFT2) {

	Gaussiankernel(kernel);

	cv::Point2f Shifts;
	Shifts.x = motionVec[imgindex].at<float>(0, 2) * rfactor;
	Shifts.y = motionVec[imgindex].at<float>(1, 2) * rfactor;

	A[imgindex] = ComposeSystemMatrix(Src, Dest, Shifts, rfactor, kernel, DMatrix,
			HMatrix, MMatrix);

	Normalization(A[imgindex], A[imgindex]);

	B[imgindex] = sparseMatSq(A[imgindex]);

	AT[imgindex] = A[imgindex].transpose();

	BT[imgindex] = B[imgindex].transpose();

	viennacl::compressed_matrix<float> tmp_vcl(A[imgindex].rows(), A[imgindex].cols(),
			A[imgindex].nonZeros());
	viennacl::compressed_matrix<float> tmp_vclT(AT[imgindex].rows(), AT[imgindex].cols(),
			AT[imgindex].nonZeros());

	viennacl::copy(A[imgindex], tmp_vcl);
	viennacl::copy(AT[imgindex], tmp_vclT);

	DHF.push_back(tmp_vcl);
	DHFT.push_back(tmp_vclT);

	viennacl::copy(B[imgindex], tmp_vcl);
	viennacl::copy(BT[imgindex], tmp_vclT);

	DHF2.push_back(tmp_vcl);
	DHFT2.push_back(tmp_vclT);

}

void cvMatToviennaVec(cv::Mat& mat, viennacl::vector<float>& out)
{
	std::vector<float> vec;
	if (mat.isContinuous()) {
		vec.assign((float*) mat.datastart, (float*) mat.dataend);
	} else {
		for (int i = 0; i < mat.rows; ++i) {
			vec.insert(vec.end(), mat.ptr<float>(i), mat.ptr<float>(i) + mat.cols);
		}
	}
	std::cout << "Vector Dimension: " << vec.size() << std::endl;
	viennacl::copy(vec, out);
}

void cvMatToviennaMat(cv::Mat& mat, viennacl::vector<float>& out, int n)
{
	std::vector<float> vec;
	if (mat.isContinuous()) {
		vec.assign((float*) mat.datastart, (float*) mat.dataend);
	} else {
		for (int i = 0; i < mat.rows; ++i) {
			vec.insert(vec.end(), mat.ptr<float>(i), mat.ptr<float>(i) + mat.cols);
		}
	}
	//TESTING
	//-------------------------------
//	std::cout << "Fy_2=" << std::endl;
//	for(int i = 0; i<50; i++)
//		{
//		std::cout << vec[i] << std::endl;
//		}
	//-------------------------------

//	Eigen::VectorXf eigVec(n);
//	for(int i = 0; i<n; i++)
//	{
//		eigVec(i) = vec[i];
//	}
	viennacl::copy(vec, out);
//	std::cout << "end of func=" << std::endl;

}

void copy_data(std::vector<cv::Mat>& Src,std::vector<float>& y0_vec,
		std::vector<float>& y1_vec, std::vector<float>& y2_vec,
		std::vector<float>& y3_vec, viennacl::vector<float>& y0,
		viennacl::vector<float>& y1, viennacl::vector<float>& y2,
		viennacl::vector<float>& y3, cv::Mat& dest,
		std::vector<float>& x_vec, viennacl::vector<float>& x)
{
	// Copy Src ->yi
		if (Src[0].isContinuous()) {
			y0_vec.assign((float*) Src[0].datastart, (float*) Src[0].dataend);
		} else {
			for (int i = 0; i < Src[0].rows; ++i) {
				y0_vec.insert(y0_vec.end(), Src[0].ptr<float>(i),
						Src[0].ptr<float>(i) + Src[0].cols);
			}
		}
//		std::cout << "y0vec dimensions: " << y0_vec.size() << std::endl;
		viennacl::copy(y0_vec, y0);

		if (Src[1].isContinuous()) {
			y1_vec.assign((float*) Src[1].datastart, (float*) Src[1].dataend);
		} else {
			for (int i = 1; i < Src[1].rows; ++i) {
				y1_vec.insert(y1_vec.end(), Src[1].ptr<float>(i),
						Src[1].ptr<float>(i) + Src[1].cols);
			}
		}
//		std::cout << "y1vec dimensions: " << y1_vec.size() << std::endl;
		viennacl::copy(y1_vec, y1);

		if (Src[2].isContinuous()) {
			y2_vec.assign((float*) Src[2].datastart, (float*) Src[2].dataend);
		} else {
			for (int i = 2; i < Src[2].rows; ++i) {
				y2_vec.insert(y2_vec.end(), Src[2].ptr<float>(i),
						Src[2].ptr<float>(i) + Src[2].cols);
			}
		}
//		std::cout << "y2vec dimensions: " << y2_vec.size() << std::endl;
		viennacl::copy(y2_vec, y2);

		if (Src[3].isContinuous()) {
			y3_vec.assign((float*) Src[3].datastart, (float*) Src[3].dataend);
		} else {
			for (int i = 3; i < Src[3].rows; ++i) {
				y3_vec.insert(y3_vec.end(), Src[3].ptr<float>(i),
						Src[3].ptr<float>(i) + Src[3].cols);
			}
		}
//		std::cout << "y3vec dimensions: " << y3_vec.size() << std::endl;
		viennacl::copy(y3_vec, y3);

		// Copy dest->x
		if (dest.isContinuous()) {
			x_vec.assign((float*) dest.datastart, (float*) dest.dataend);
		} else {
			for (int i = 0; i < dest.rows; ++i) {
				x_vec.insert(x_vec.end(), dest.ptr<float>(i),
						dest.ptr<float>(i) + dest.cols);
			}
		}
//		std::cout << "x vec dimensions: " << x_vec.size() << std::endl;
		viennacl::copy(x_vec, x);

}

Eigen::SparseMatrix<float, Eigen::RowMajor, int> init_xshiftMatrix(int n, int row_size, int x_shift, int is_circular)
{
	Eigen::SparseMatrix<float,Eigen::RowMajor, int> Ti(n, n);
	std::vector<int> one_indeces(row_size);
	int index, one_index = 0;
	int row_count = 0;
	//std::cout << "row_size=" << row_size << std::endl;

	// calculate one_indeces vector
	for (int i = 0; i < row_size; i++)
	{
		//std::cout << "i=" << i << std::endl;

		if(x_shift<0)
		{
			index = i+abs(x_shift);
			//std::cout << "index=" << index << std::endl;
			if(index<row_size)
				one_indeces[i] = index;
			else
				if(is_circular==1)
					one_indeces[i] = index-row_size;
				else
					one_indeces[i] = -1;
		}
		else
		{
			index = i-x_shift;
			//std::cout << "index=" << index << std::endl;
			if(index>=0)
				one_indeces[i] = index;
			else
				if(is_circular==1)
					one_indeces[i] = index+row_size;
				else
					one_indeces[i] = -1;
		}
	}
	for (int k = 0; k < n; k++)
	{
		if(k%row_size==0 && k!=0)
		{
			row_count++;
			//std::cout << "row_count=" << row_count << std::endl;
		}
		one_index = one_indeces[k%row_size];
		index = row_count*row_size + one_index; // offset + index of one
		if(is_circular!=1 && one_index==-1)
		{
			//std::cout << "continue" << std::endl;
			continue;
		}
		Ti.insert(k, index) = 1;

	}
	//std::cout << Ti << std::endl;
	Ti.makeCompressed();
	return Ti;
}

Eigen::SparseMatrix<float, Eigen::RowMajor, int> init_yshiftMatrix(int n, int row_size, int y_shift)
{
	Eigen::SparseMatrix<float,Eigen::RowMajor, int> Ti(n, n);
	//upper shift
	if(y_shift < 0) {
		int row_iterator =0;
		for(int col_iterator = -y_shift*row_size; col_iterator<n; col_iterator++) {
			Ti.insert(row_iterator, col_iterator ) = 1;
			row_iterator++;
		}
	}
	//lower shift
	else {
		int col_iterator =0;
		for(int row_iterator = y_shift*row_size; row_iterator<n; row_iterator++) {
			Ti.insert(row_iterator, col_iterator ) = 1;
			col_iterator++;
		}
	}

	//std::cout << Ti << std::endl;
	Ti.makeCompressed();
	return Ti;
}

float getPSNR(cv::Mat& I1, cv::Mat& I2)
{
	cv::Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32FC1);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         // sum elements per channel

    float sse = s.val[0]; // sum channels

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
    	float mse = sse /(float)(I1.cols*I1.rows);
    	float maxValue = pow(2,16)-1;
    	float psnr = 10.0*log10(pow(maxValue,2)/mse);
        return psnr;
    }
}


viennacl::vector<float> hadamard_vector_vector (viennacl::vector<float>& v1, viennacl::vector<float>& v2)
{
	return element_prod(v1, v2);
}


/********************************************************************************************************************************************************************************/
/******************************************** ADAM  Method*********************************************************************************************************************/
/********************************************************************************************************************************************************************************/


/*
 * function to calculate the partial derivative of Ui(zi) : slide 20 (eq. 17)
 */
Eigen::VectorXf PD_Ui(Eigen::VectorXf & z_i, Eigen::VectorXf & x, Eigen::VectorXf & p,
		float rho, int I0, int n)
{
	return (rho*z_i - rho*x - p);
}


/*
 * function to calculate the partial derivative of Gi(zi) : slide 20
 */
viennacl::vector<float> PD_Gi(viennacl::vector<float>& z_i_gpu,
				viennacl::compressed_matrix<float>& AT_i_gpu,
				viennacl::compressed_matrix<float>& BT_i_gpu,
				viennacl::vector<float>& neg_exp_z_i_gpu,
				viennacl::vector<float>& numerator_gpu,
				viennacl::vector<float>& Mi_gpu,
				int I0, int n, int m){

	// common numerator  = y_i -A_i*I0*exp(-z_i)
	// common denominator  = B_i*I0*exp(-z_i) + sigma^2
	// Mi gpu = 1/denominator

	viennacl::vector<float> exp1 = I0*neg_exp_z_i_gpu;
	viennacl::vector<float> mi_gpu = hadamard_vector_vector(numerator_gpu, Mi_gpu);

	viennacl::vector<float> ATM = prod(AT_i_gpu,mi_gpu);
	viennacl::vector<float> pd1 = hadamard_vector_vector(ATM, exp1);

	viennacl::vector<float> BTM_tmp = hadamard_vector_vector(mi_gpu, mi_gpu);
	viennacl::vector<float> BTM = prod(BT_i_gpu,BTM_tmp);
	viennacl::vector<float> pd2 = hadamard_vector_vector(BTM, exp1);
	pd2 = 0.5*pd2;

	return pd1 + pd2;
}

/*
 * function to calculate the partial derivative of Hi(zi) : slide 21
 */
viennacl::vector<float> PD_Hi(viennacl::vector<float>& z_i_gpu,
						viennacl::compressed_matrix<float>& BT_i_gpu,
						viennacl::vector<float>& neg_exp_z_i_gpu,
						viennacl::vector<float>& Mi_gpu,
						int I0){

	viennacl::vector<float> exp1 = I0*neg_exp_z_i_gpu;

	viennacl::vector<float> tmp = prod(BT_i_gpu,Mi_gpu);
	viennacl::vector<float> pdhi = hadamard_vector_vector(tmp, exp1);

	return -0.5*pdhi;
}


/*
 * ADAM's method function
 */
void ADAM(Eigen::VectorXf & z_i, viennacl::vector<float>& z_i_gpu,
		Eigen::MatrixXf & A_i, viennacl::compressed_matrix<float> & A_i_gpu,
		Eigen::MatrixXf & AT_i, viennacl::compressed_matrix<float> & AT_i_gpu,
		Eigen::MatrixXf & B_i, viennacl::compressed_matrix<float> & B_i_gpu,
		Eigen::MatrixXf & BT_i, viennacl::compressed_matrix<float> & BT_i_gpu,
				Eigen::VectorXf & y_i,  viennacl::vector<float>& y_i_gpu,
				Eigen::VectorXf & sigma_sq,
				Eigen::SparseMatrix<float, Eigen::RowMajor, int> & T_i,
				Eigen::VectorXf & X_i, Eigen::VectorXf & P_i ,float rho , Eigen::VectorXf & rhoEig,int I0,
				int n, int m, int iter){

	// --------------------------------------------- //
	// 			Initialize ADAM parameters 			//
	// --------------------------------------------- //
	float stepsize = 0.1;
	float beta1 = 0.5;
	float beta2 = 0.999;
	viennacl::vector<float> epsilon = viennacl::scalar_vector<float>(n, 1e-8f);

	// Adaptive Step Size
	if (iter%5 == 0)
	{
		stepsize /= 2;
	}


	// --------------------------------------------- //
	// Initialize the partial derivatives parameters //
	// --------------------------------------------- //
	viennacl::vector<float> sigma_sq_gpu(m);
	copy(sigma_sq, sigma_sq_gpu);

	Eigen::VectorXf neg_exp_z_i(n);
	viennacl::vector<float> neg_exp_z_i_gpu(n);

	//exp(-z_i)
	neg_exp_z_i_gpu = element_exp(-1*z_i_gpu);
	copy(neg_exp_z_i_gpu,neg_exp_z_i);

	// common numerator  = y_i -A_i*I0*exp(-z_i)
	viennacl::vector<float> numerator_gpu = y_i_gpu - I0*prod(A_i_gpu,neg_exp_z_i_gpu);

	// common denominator  = B_i*I0*exp(-z_i) + sigma^2
	viennacl::vector<float> denominator_gpu = I0*prod(B_i_gpu,neg_exp_z_i_gpu) + sigma_sq_gpu;


	viennacl::vector<float> m1_gpu = viennacl::scalar_vector<float>(m, 1.0f); // vector of ones
	viennacl::vector<float> Mi_gpu = element_div(m1_gpu,denominator_gpu); // 1/denominator

	// -------------------------------------------------------------------------------------- //
	// 										 ADAM Algorithm 								  //
	// -------------------------------------------------------------------------------------- //

	// ------------------------------------------ //
	// 	Initialize partial derivative vectors 	  //
	// ------------------------------------------ //
	viennacl::vector<float> pdgi = viennacl::scalar_vector<float>(n, 0.0f);
	viennacl::vector<float> pdhi = viennacl::scalar_vector<float>(n, 0.0f);
	viennacl::vector<float> sum_pd = viennacl::scalar_vector<float>(n, 0.0f);;
	viennacl::vector<float> firstMoment_gpu = viennacl::scalar_vector<float>(n, 0.0f);
	viennacl::vector<float> secondMoment_gpu = viennacl::scalar_vector<float>(n, 0.0f);
	viennacl::vector<float> tmp1 = viennacl::scalar_vector<float>(n, 0.0f);
	viennacl::vector<float> tmp2 = viennacl::scalar_vector<float>(n, 0.0f);
	viennacl::vector<float> tmp3 = viennacl::scalar_vector<float>(n, 0.0f);
	viennacl::vector<float> sum_pd_sq = viennacl::scalar_vector<float>(n, 0.0f);
	viennacl::vector<float> z_i_gpu_old = viennacl::scalar_vector<float>(n, 0.0f);

	float error = 10;

	Eigen::VectorXf sum_pd_cpu(n);
	Eigen::VectorXf pdui(n);

	int t = 0;


	while(error > 1e-12)
	{
		z_i_gpu_old = z_i_gpu;

		// 1. Update time
		t++;

		// 2. Get gradients "sum_pd" at timestep t
		pdgi = PD_Gi(z_i_gpu, AT_i_gpu, BT_i_gpu, neg_exp_z_i_gpu, numerator_gpu,Mi_gpu, I0, n, m);
		pdhi = PD_Hi(z_i_gpu, BT_i_gpu, neg_exp_z_i_gpu, Mi_gpu, I0);
		sum_pd = pdgi +pdhi;
		copy(sum_pd, sum_pd_cpu);
		pdui = PD_Ui(z_i, X_i, P_i, rho, I0, n);
		sum_pd_cpu += pdui;
		copy(sum_pd_cpu, sum_pd);

		sum_pd_sq = hadamard_vector_vector(sum_pd, sum_pd);

		//3. Update first moment
		tmp1 = beta1 * firstMoment_gpu;
		tmp2 = (1 - beta1) * sum_pd;
		firstMoment_gpu = tmp1 + tmp2;
		firstMoment_gpu = firstMoment_gpu / (1.0f - pow(beta1,t));

		//4. Update second moment
		tmp1 = beta2 * secondMoment_gpu;
		tmp2 = (1 - beta2) * sum_pd_sq;
		secondMoment_gpu = tmp1 + tmp2;
		secondMoment_gpu = secondMoment_gpu / (1.0f - pow(beta2,t));

		//5. Gradient descent
		tmp1 = stepsize * firstMoment_gpu;
		tmp2 = element_sqrt(secondMoment_gpu) + epsilon;
		tmp3 =  element_div(tmp1 , tmp2);
		z_i_gpu = z_i_gpu - tmp3;

		// CHeck for convergence
		tmp1 = z_i_gpu_old - z_i_gpu;
		error = norm_1(tmp1) / n;

	}



}

float calculate_g_i(viennacl::vector<float>& z_i_gpu, viennacl::compressed_matrix<float> & A_i_gpu,
		viennacl::compressed_matrix<float> & B_i_gpu, viennacl::vector<float>& y_i_gpu, Eigen::VectorXf & sigma_sq,
		int I0, int n, int m)
{

		viennacl::vector<float> sigma_sq_gpu(m);
		copy(sigma_sq, sigma_sq_gpu);
		viennacl::vector<float> neg_exp_z_i_gpu(n);
		Eigen::VectorXf inv_denominator_cpu(m);
		Eigen::MatrixXf W(m,m);
		viennacl::vector<float> W_num_gpu(m);
		viennacl::matrix<float> W_gpu(m,m);

		viennacl::vector<float> log_den_gpu(m);

		//exp(-z_i)
		neg_exp_z_i_gpu = element_exp(-1*z_i_gpu);


		float G_i =0;

		// y_i -A_i*I0*exp(-z_i)
		viennacl::vector<float> numerator_gpu = y_i_gpu - I0*prod(A_i_gpu,neg_exp_z_i_gpu);

		// B_i*I0*exp(-z_i) + sigma^2
		viennacl::vector<float> denominator_gpu = I0*prod(B_i_gpu,neg_exp_z_i_gpu) + sigma_sq_gpu;

		// Calculate G(i)
		viennacl::vector<float> ones_gpu = viennacl::scalar_vector<float>(m, 1.0f);
		viennacl::vector<float> inv_denominator_gpu = element_div(ones_gpu, denominator_gpu);
		copy(inv_denominator_gpu, inv_denominator_cpu);
		W = inv_denominator_cpu.asDiagonal();
		copy(W, W_gpu);
		W_num_gpu = prod(W_gpu, numerator_gpu);
		G_i = 0.5 * inner_prod(W_num_gpu, numerator_gpu);

		log_den_gpu = element_log(denominator_gpu);

		float H_i = inner_prod(log_den_gpu, ones_gpu);


		float g_i = G_i + H_i;

		return g_i;
}

/*********************************************************************************************************************************/
/*********************************************************************************************************************************/

int main(int argc, char** argv) {

	// ------------------------------------------------------------------------------------------ //
	// ----------------------------------- INITIALIZE PARAMETERS -------------------------------- //
	// ------------------------------------------------------------------------------------------ //
	int M                 = 4; 	 // M
	int w                 = 1; 	 // window size, can be tuned but odd number
	int rfactor           = 2; 	 //magnification factor
	float psfWidth        = 3;   // ?
	int i_uBound          = M + pow((2*w+1),2);
	int iterate_flag      = 1;   // 1->iterate, 0-> don't do anything(used to run test codes)

	int I_0 			  = 60000; 			//6e4 250
	float alpha 		  = 0.4; 			// alpha in matrix gamma, can be tuned
	float beta 			  = 0.0001; 		// needs to be tuned
	float sigma 		  = 500 ; 			// sigma can be 100, 150, 200,...500
	float rho 			  = 1;   			// 0.1 -> 10
	int n_iterations      = 50;  			// ADMM number of iterations


	// ------------------------------------------------------------------------------------------ //
	// --------------------------------- SAVE ARGUMENTS IN LOG FILE----------------------------- //
	// ------------------------------------------------------------------------------------------ //
	log("\n");
	log("Parameters for process : "+ boost::lexical_cast <std::string >(::getpid()) +"\n");
	log("I0:"+ boost::lexical_cast <std::string >(I_0)+"\n");
	log("alpha:"+ boost::lexical_cast <std::string >(alpha)+"\n");
	log("beta:"+ boost::lexical_cast <std::string >(beta)+"\n");
	log("sigma:"+ boost::lexical_cast <std::string >(sigma)+"\n");
	log("rho:"+ boost::lexical_cast <std::string >(rho)+"\n");
	log_cost("\n");
	log_cost("Parameters for process : "+ boost::lexical_cast <std::string >(::getpid()) +"\n");
	log_cost("I0:"+ boost::lexical_cast <std::string >(I_0)+"\n");
	log_cost("alpha:"+ boost::lexical_cast <std::string >(alpha)+"\n");
	log_cost("beta:"+ boost::lexical_cast <std::string >(beta)+"\n");
	log_cost("sigma:"+ boost::lexical_cast <std::string >(sigma)+"\n");
	log_cost("rho:"+ boost::lexical_cast <std::string >(rho)+"\n");

	// ------------------------------------------------------------------------------------------ //
	// --------------------------------- CUSTOM KERNEL DECLARATION ------------------------------ //
	// ------------------------------------------------------------------------------------------ //
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId] (), 0, 0 };
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);

	// Get the first device of the context
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "Kernel.cl");
	OpenCL::buildProgram(program, devices);// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages

	// Create a kernel object
	cl::Kernel ProximalOperatorKernel (program, "ProximalOperatorKernel");
	cl::Kernel ProximalOperatorKernel2 (program, "ProximalOperatorKernel2"); // max(X_k+ P/rho, 0)

	// Declare some valuesADAM
	std::size_t wgSize = 256; // Number of work items per work group
	std::size_t count = wgSize * 100; // Overall number of work items = Number of elements
	std::size_t size = count * sizeof (float); // Size of data in bytes

	// Allocate space for input data and for output data from CPU and GPU on the host
	Eigen::VectorXf h_outputGpu (count);

	// Allocate space for input and output data on the device
	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, size);

	// ------------------------------------------------------------------------------------------ //
	// ------------------------------------------------------------------------------------------ //


	// Declarations
	std::vector<cv::Mat> Src(M);
	cv::Mat dest, gnd_truth;
	cv::Mat f_y2_mat;
	cv::Mat kernel = cv::Mat::zeros(cv::Size(psfWidth, psfWidth), CV_32F);
	viennacl::vector<float> y0, y1, y2, y3; // input images
	viennacl::vector<float> x; // output image
	viennacl::vector<float> f_y2; // Fbic(y2)
	std::vector<viennacl::compressed_matrix<float> > DHF;   //Ai
	std::vector<viennacl::compressed_matrix<float> > DHFT;  //ATi
	std::vector<viennacl::compressed_matrix<float> > DHF2;  //Bi
	std::vector<viennacl::compressed_matrix<float> > DHFT2; //BTi
	Eigen::SparseMatrix<float, Eigen::RowMajor, int> DMatrix;
	Eigen::SparseMatrix<float, Eigen::RowMajor, int> HMatrix;
	Eigen::SparseMatrix<float, Eigen::RowMajor, int> MMatrix;  // shift matrix
	std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor, int> > AT(M); // transpose of matrix A_i
	std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor, int> > A(M);  // matrix A_i
	std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor, int> > BT(M); //transpose of matrix B_i
	std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor, int> > B(M);  // matrix B_i


	/***** Generate motion parameters & load source images ******/
	std::vector<cv::Mat> motionvec;

	motionMat(motionvec, M, rfactor, true);
	for (int i = 0; i < M; i++)
	{
		Src[i] = cv::imread("../Images/PCB/LR" + boost::lexical_cast<std::string>(i + 1) + ".tif", CV_LOAD_IMAGE_ANYDEPTH  );


		Src[i].convertTo(Src[i], CV_32FC1);


		//TODO: should be out of the loop
		dest = cv::Mat(Src[0].rows * rfactor, Src[0].cols * rfactor, CV_32FC1 );
		cv::resize(Src[0], dest, dest.size(), 0, 0, INTER_CUBIC); // Slide 26

		/***** Generate Matrices A = DHF, inverse A = DHFT and B = DHF2, inverse B = DHFT2 ******/
		GenerateAT(Src[i], dest, i, motionvec, kernel, rfactor, DMatrix,
				HMatrix, MMatrix, A, AT, B, BT, DHF, DHFT, DHF2, DHFT2);
	}

	gnd_truth = cv::imread("../Images/PCB/origin.tif", CV_LOAD_IMAGE_ANYDEPTH);
	gnd_truth.convertTo(gnd_truth, CV_32FC1);

	// unfold matrices (images) and copy to viennacl vectors
	std::vector<float> y0_vec, y1_vec, y2_vec, y3_vec;
	std::vector<float> x_vec;
	copy_data(Src, y0_vec, y1_vec, y2_vec, y3_vec,y0, y1, y2, y3, dest, x_vec, x);
	std::vector<viennacl::vector<float> > y_viennacl(4);
	y_viennacl[0] = y0;
	y_viennacl[1] = y1;
	y_viennacl[2] = y2;
	y_viennacl[3] = y3;


	//Init x // fixme: USELESS
	//x = -1*element_log((x/I_0));
	x.clear();

	// Set m & n
	const int m = y0.size(); 	// input image size (as a vector)
	const int n = x.size();	    // output image size (as a vector)
	int x_row_size = dest.rows; // output image row size
	int x_col_size = dest.cols; // output image col size
	//std::cout << "m= " << m << ", n= " << n<< std::endl;

	// Calculate Fbic(y2) fixme:: which y?
	f_y2_mat = cv::Mat(Src[0].rows * rfactor, Src[0].cols * rfactor, CV_32FC1); //fixme CV_16UC1
	cv::resize(Src[0], f_y2_mat, f_y2_mat.size(), 0, 0, INTER_CUBIC); // Slide 26
	f_y2_mat.convertTo(f_y2_mat, CV_16UC1);
	int write_bool = imwrite("../bicubic.tif", f_y2_mat);
	f_y2_mat.convertTo(f_y2_mat, CV_32FC1);

	std::string text = "PSNR(bicubic & ground truth):" + boost::lexical_cast <std::string >(getPSNR(f_y2_mat, gnd_truth)) + "\n";
	log(text);
	cvMatToviennaMat(f_y2_mat, f_y2, n);

	// Initialize Identity Matrix
	Eigen::SparseMatrix<float, Eigen::RowMajor, int> I(n,n);
	I.setIdentity();
	viennacl::compressed_matrix<float> Identity;
	viennacl::copy( I, Identity);

	//Initialize p vector (Eigen)
	std::vector<Eigen::VectorXf> p_k(i_uBound);

	//Initialize z vector (viennacl) and convert to Eigen
	std::vector<Eigen::VectorXf> z_k(i_uBound);
	std::vector<viennacl::vector<float> > z_k_viennacl(i_uBound, viennacl::vector<float>(n));
	for (int i = 0; i < i_uBound; i++) {
		z_k_viennacl[i]= -1*element_log(f_y2/I_0);
		//z_k_viennacl[i]= f_y2; // fixme: initialize to image 2 directly
		z_k[i] = Eigen::VectorXf(n);
		copy(z_k_viennacl[i], z_k[i]);

		// initialize p_k[i]
		p_k[i] = Eigen::VectorXf(n);
		p_k[i].setZero();
	}

	// ------------------------------------------------------------------------------------------ //
	// ---------------------------------- ADAM method Initializations -------------------------- //
	// ------------------------------------------------------------------------------------------ //
	//Copy A & B matrices from Eigen sparse type to MatrixXf type
	//TODO (can be during initialization)
	std::vector<Eigen::MatrixXf> A_i(M), AT_i(M), B_i(M), BT_i(M);
		std::vector<Eigen::VectorXf> Y(M); 	// Copy y images into Eigen data type //TODO (can be during initialization)
		for(int i = 0; i < M; i++)
		{
			A_i[i]  = A[i]; // from sparse to matrixXf
			AT_i[i] = AT[i];
			B_i[i]  = B[i];
			BT_i[i] = BT[i];

			//initialize Y[i]
			Y[i] = Eigen::VectorXf(m);
		}
		copy(y_viennacl[1] ,Y[1]);
		copy(y_viennacl[2] ,Y[2]);
		copy(y_viennacl[3] ,Y[3]);
		Eigen::VectorXf rhoEig(n), sigmasqEig(m);
		rhoEig.setOnes();
		sigmasqEig.setOnes();
		rhoEig *= rho;
		sigmasqEig *= sigma*sigma;
	// ------------------------------------------------------------------------------------------ //
	// ------------------------------------------------------------------------------------------ //


	// ------------------------------------------------------------------------------------------ //
	// ------------------------- Calculate shift matrices (S_x)^q (S_y)^p ----------------------- //
	// ------------------------------------------------------------------------------------------ //
	Eigen::SparseMatrix<float, Eigen::RowMajor, int> x_shift_one = init_xshiftMatrix(n, x_row_size, 1, 0);
	Eigen::SparseMatrix<float, Eigen::RowMajor, int> x_shift_neg_one = init_xshiftMatrix(n, x_row_size, -1, 0);
	Eigen::SparseMatrix<float, Eigen::RowMajor, int> y_shift_one = init_yshiftMatrix(n, x_row_size, 1);
	Eigen::SparseMatrix<float, Eigen::RowMajor, int> y_shift_neg_one = init_yshiftMatrix(n, x_row_size, -1);
	std::vector<float> gamma_pq;
	std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor, int> > T; //T --> Slide 13
	for(int q=-w;q<=w;q++) // x shift
	{
		Eigen::SparseMatrix<float, Eigen::RowMajor, int> x_shift_q = I;
		Eigen::SparseMatrix<float, Eigen::RowMajor, int> x_temp;

		if(q<0) // initialize x_shift_q
		{
			x_temp = x_shift_neg_one;
			x_shift_q = x_shift_neg_one;
		}
		else
			if(q>0)
			{
				x_temp = x_shift_one;
				x_shift_q = x_shift_one;
			}

		if(abs(q)>1) // calculate (S_x)^q
		{
			for(int i=1; i<abs(q); i++)
			{
				x_shift_q *=  x_temp;
			}
		}

		int abs_q = abs(q);
		for(int p=-w;p<=w;p++) //y shift
		{
			Eigen::SparseMatrix<float, Eigen::RowMajor, int> y_shift_p = I;
			Eigen::SparseMatrix<float, Eigen::RowMajor, int> y_temp;

			if(p<0) // initialize x_shift_q
			{
				y_temp = y_shift_neg_one;
				y_shift_p = y_shift_neg_one;
			}
			else
				if(p>0)
				{
					y_temp = y_shift_one;
					y_shift_p = y_shift_one;
				}

			if(abs(p)>1) // calculate (S_y)^p
			{
				for(int i=1; i<abs(p); i++)
				{
					y_shift_p *=  y_temp;
				}
			}

			Eigen::SparseMatrix<float, Eigen::RowMajor, int> prod_result = x_shift_q  * y_shift_p;
			Eigen::SparseMatrix<float, Eigen::RowMajor, int> result = I - prod_result;
			if(!(q==0 && p==0))
			{
				T.push_back(result);
				int abs_p = abs(p);
				gamma_pq.push_back( pow(alpha,abs_p + abs_q));
			}
		}
	}


	// ------------------------------------------------------------------------------------------ //
	// ------------------------------- Start ADMM iterations ------------------------------------ //
	// ------------------------------------------------------------------------------------------ //
	float energy_func = 0;
	std::vector<float> g_i(i_uBound);

	if(iterate_flag==1)
	{
		// START ITERATIONS
		Eigen::VectorXf x_k(n); // output image vector
		for(int iter = 0; iter<n_iterations; iter++)
		{

			std::cout << "----------------------" << std::endl;
			std::cout << "Iteration: " << iter << std::endl;
			int c_iteration_start=clock();
			// ------------------------------------------------------------------------------------------ //
			// --------------------------------- Update X^(k+1) ----------------------------------------- //
			// ------------------------------------------------------------------------------------------ //
			// Conjugate Gradient Step: solve Ax=b
			Eigen::SparseMatrix<float, Eigen::RowMajor, int> conj_grad_A(n,n);
			Eigen::VectorXf conj_grad_b(n);
			conj_grad_A.setZero();
			conj_grad_b.setZero();

			// find argmin x: calculate A,b
			for (int i = 0;  i < i_uBound; i++) {
				if(i < M || i == i_uBound - 1) // Ti = Identity matrix (not used)
				{
					conj_grad_A += rho*I;
					conj_grad_b += rho*z_k[i] - p_k[i];
				}
				else
				{
					Eigen::SparseMatrix<float, Eigen::RowMajor, int> tmp = rho*T[i-M].transpose()*T[i-M];
					conj_grad_A += tmp;
					Eigen::VectorXf tmp2 = (rho*z_k[i] - p_k[i]);
					Eigen::VectorXf tmp3 = T[i-M].transpose()*tmp2;
					conj_grad_b += tmp3;
				}
			}
			int c_xstep_start=clock();
			x_k = viennacl::linalg::solve(conj_grad_A, conj_grad_b, viennacl::linalg::cg_tag());
			int c_xstep_stop=clock();
			std::cout << "x update time = " << (c_xstep_stop-c_xstep_start)/double(CLOCKS_PER_SEC) << std::endl;

			// TESTING:: Print some x values
			std::cout << "x update -> done"  << std::endl;
			for(int j=0; j<5; j++)
			{
				std::cout << "x_k="  << x_k.coeff(j) << std::endl;
			}

			// ------------------------------------------------------------------------------------------ //
			// ---------------------------------Update Z_i^(k+1) ---------------------------------------- //
			// ------------------------------------------------------------------------------------------ //
			int c_zstep_start=clock();
			Eigen::VectorXf h_input, tmp;
			for (int i = 0;  i < i_uBound; i++) {
				if(i < M) // ADAM Step Update
				{
					ADAM(z_k[i] , z_k_viennacl[i],
							A_i[i], DHF[i],
							AT_i[i], DHFT[i],
							B_i[i],DHF2[i],
							BT_i[i], DHFT2[i],
							Y[i], y_viennacl[i],
							sigmasqEig, T[i], x_k, p_k[i], rho, rhoEig, I_0, n, m, iter);

				}
				else // Proximal Operator Update
					if(i<i_uBound-1) // slide 24
					{
						tmp = p_k[i]/rho;
						h_input =  x_k + tmp;
						float threshold = gamma_pq[i-M]*beta*(1/rho);

						// setup Kernel data
						memset(h_outputGpu.data(), 255, size);
						queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
						queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());

						// Launch kernel on the device
						ProximalOperatorKernel.setArg< cl::Buffer >(0, d_input);
						ProximalOperatorKernel.setArg< cl::Buffer >(1, d_output);
						ProximalOperatorKernel.setArg< cl_float>(2, threshold);
						cl::Event eventGPU;
						queue.enqueueNDRangeKernel(ProximalOperatorKernel,cl::NullRange , count, wgSize,NULL,&eventGPU);
						queue.finish();

						// Copy output data back to host
						cl::Event eventRB;
						queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(),NULL, &eventRB);
						z_k[i] = h_outputGpu; // Update z_k+1

					}
					else // slide 25
					{
						tmp = p_k[i]/rho;
						h_input = x_k + tmp;

						// setup Kernel data
						memset(h_outputGpu.data(), 255, size);
						queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
						queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());

						// Launch kernel on the device
						ProximalOperatorKernel2.setArg< cl::Buffer >(0, d_input);
						ProximalOperatorKernel2.setArg< cl::Buffer >(1, d_output);
						cl::Event eventGPU;
						queue.enqueueNDRangeKernel(ProximalOperatorKernel2,cl::NullRange , count, wgSize,NULL,&eventGPU);
						queue.finish();

						// Copy output data back to host
						cl::Event eventRB;
						queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(),NULL, &eventRB);

						z_k[i] = h_outputGpu; // Update z_k+1


					}

				copy(z_k[i], z_k_viennacl[i]); // copy to viennacl (both are consistent)
			}
			//std::cout << "z update -> done"  << std::endl;
			int c_zstep_stop=clock();
			std::cout << "z update time = " << (c_zstep_stop-c_zstep_start)/double(CLOCKS_PER_SEC) << std::endl;


			// ------------------------------------------------------------------------------------------ //
			// ---------------------------------Update P_i^(k+1) ---------------------------------------- //
			// ------------------------------------------------------------------------------------------ //
			int c_pstep_start=clock();
			viennacl::vector<float> x_k_vienna(n), p_k_vienna(n), rho_vec(n), shifted_x_k_vienna(n);
			viennacl::copy(rhoEig, rho_vec);
			viennacl::copy(x_k, x_k_vienna);
			viennacl::compressed_matrix<float> T_gpu(n,n);
			for (int i = 0;  i < i_uBound; i++) {
					if(i < M || i==i_uBound-1)
					{
						viennacl::copy(p_k[i], p_k_vienna);
						viennacl::vector<float> tmp  =  x_k_vienna - z_k_viennacl[i];
						viennacl::vector<float> tmp2 = rho*tmp; //element_prod(tmp, rho_vec)
						p_k_vienna += tmp2;
						viennacl::copy(p_k_vienna, p_k[i]);
					}
					else
						if(i<i_uBound-1)
						{
							copy(T[i-M], T_gpu);
							Eigen::VectorXf tmp(n);
							viennacl::vector<float> tmp_gpu = prod(T_gpu,x_k_vienna);
							copy(tmp_gpu, tmp);
							tmp = tmp - z_k[i];
							tmp *= rho;
							p_k[i] += tmp;
						}

			}

			int c_pstep_stop=clock();
			std::cout << "update step p: " << (c_pstep_stop-c_pstep_start)/double(CLOCKS_PER_SEC) << std::endl;

			int c_iteration_stop=clock();
			std::cout << "Iteration time = " << (c_iteration_stop-c_iteration_start)/double(CLOCKS_PER_SEC) << std::endl;



			// ------------------------------------------------------------------------------------------ //
			// --------------------------------- Print Cost Function ----------------------------------------- //
			// ------------------------------------------------------------------------------------------ //
			energy_func = 0;
			viennacl::compressed_matrix<float> T_gpu_tmp(n,n);
			for(int i=0; i<i_uBound;i++)
			{


				if(i < M )
				{
					g_i[i] = calculate_g_i(x_k_vienna,DHF[i], DHF2[i],y_viennacl[i], sigmasqEig, I_0, n, m);

				}
				else if (i < i_uBound - 1)
				{
					copy(T[i-M], T_gpu_tmp);
					shifted_x_k_vienna = prod(T_gpu_tmp,x_k_vienna);
					g_i [i] = gamma_pq[i-M]*beta * norm_1(shifted_x_k_vienna);

				}
				else
				{
					g_i[i] = 0;
				}


			}

			for(int i=0; i<i_uBound;i++)
				energy_func += g_i[i];



			std::cout<< "J = " << energy_func << std::endl;
			log_cost("J:"+ boost::lexical_cast <std::string >(energy_func)+"\n");

			// Write Output Image
			//-----------------------//
			x_k_vienna = I_0*element_exp(-1*x_k_vienna);
			Eigen::VectorXf x_k_scaledUp(n);
			copy(x_k_vienna, x_k_scaledUp);

			std::string imageName = "../output_imgs/output_adam" + boost::lexical_cast<std::string> (iter) + ".tif";

			dest = cv::Mat(x_row_size, x_col_size, CV_32FC1, x_k_scaledUp.data());
			std::cout << "write image"  << std::endl;
			dest.convertTo(dest, CV_16UC1);
			imwrite(imageName, dest);
			dest.convertTo(dest, CV_32FC1);

			float psnr = getPSNR(gnd_truth, dest);
			std::cout << "PSNR iteration " << iter << " = " << psnr << std::endl;
			text = "PSNR iteration "+boost::lexical_cast<std::string>(iter)+": "+boost::lexical_cast<std::string>(psnr)+"\n";
			log(text);
			std::cout << "----------------------" << std::endl;
		}

	} // end of ADMM iteration loop

	return 0;
}
