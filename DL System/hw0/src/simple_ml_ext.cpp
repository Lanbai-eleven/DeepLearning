#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

float* matrix_dot(const float *A, const float *B, size_t index, size_t m, 
                size_t n, size_t k, bool transpose_A=false)
{
    size_t cur_m = (transpose_A? n:m);
    size_t cur_n = (transpose_A? m:n);
    float *result = new float[cur_m*k];
    for(size_t i=0; i<cur_m; i++){
        for(size_t j=0; j<k; j++){
            result[i*k+j] = 0;
            for(size_t l=0; l<cur_n; l++){
                if(transpose_A)
                    result[i*k+j] += A[index*n+l*n+i] * B[l*k+j];
                else
                    result[i*k+j] += A[index*n+i*n+l] * B[l*k+j];
            }
        }
    }
    return result;
}

float* softmax(const float *logits, size_t m, size_t k)
{
    float *result = new float[m*k];
    for(size_t i=0; i<m; i++){
        float sum = 0;
        for(size_t j=0; j<k; j++){
            result[i*k+j] = exp(logits[i*k+j]);
            sum += result[i*k+j];
        }
        for(size_t j=0; j<k; j++){
            result[i*k+j] /= sum;
        }
    }
    return result;
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */
    //batch
    for(size_t index=0; index<m; index+=batch){
        //logits
        float *logits = matrix_dot(X, theta, index, batch, n, k);
        //softmax
        float *softmax_result = softmax(logits, batch, k);
        //gradient
        for(size_t i = 0; i < batch; i++) {
            softmax_result[i*k + (uint8_t)y[i+index]] -= 1.0;
    }
        auto gradient = matrix_dot(X, softmax_result, index, batch, n, k, true);
        //update
        for(size_t i=0; i<n; i++){
            for(size_t j=0; j<k; j++){
                theta[i*k+j] -= lr * gradient[i*k+j]/((m-index)<batch? m-index:batch);
            }
        }
        // //delete
        delete[] logits;
        delete[] softmax_result;
        delete[] gradient;
    }

}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
