#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>

#define MATRIX_WIDTH 1000
#define MATRIX_LENGTH (MATRIX_WIDTH * MATRIX_WIDTH)

int main() {

    // Init matrices
    std::vector<float> h_a(MATRIX_LENGTH), h_b(MATRIX_LENGTH), h_c(MATRIX_LENGTH);

    for (int i = 0; i < MATRIX_LENGTH; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    float *output = new float[INT32_MAX];

    std::cout << "Matrix A:\n";
    // for (int i = 0; i < h_a.size(); i++) {
    //     std::cout << h_a[i] << " ";
    // }
    
    std::cout << "Matrix B:\n";
    // for (int i = 0; i < h_b.size(); i++) {
    //     std::cout << h_b[i] << " ";
    // }

    // File stuff - convert file to string
    std::fstream t("cl_kernel.cl");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string cl_code = buffer.str();

    std::cout << "CL code: " << cl_code << "\n";

    // Find and pick device and platform
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    cl::Platform default_platform = all_platforms[0];

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    cl::Device default_device = all_devices[0];

    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    cl::Program::Sources sources;
    sources.push_back({cl_code.c_str(), cl_code.length()});

    cl::Context context(CL_DEVICE_TYPE_DEFAULT);

    // Build program, catch error if it doesn't build
    cl::Program program(context, sources);
    try {
        program.build({default_device});
    }
    catch (cl::Error& e){
        std::cout << "Build error:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }
    
    cl::CommandQueue queue(context);
    // cl::Program program(context, cl_code.c_str(), true);

    cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
    cl::Buffer d_b(context, h_b.begin(), h_b.end(), true);
    cl::Buffer d_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * MATRIX_LENGTH);

    // cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> vec_add(program, "vec_add");

    std::cout << "Starting memory copy\n";
    queue.enqueueWriteBuffer(d_a, CL_TRUE, 0, sizeof(float) * MATRIX_LENGTH, h_a.data());
    queue.enqueueWriteBuffer(d_b, CL_TRUE, 0, sizeof(float) * MATRIX_LENGTH, h_b.data());

    cl::Kernel kernel(program, "matrixVectorMul");

    kernel.setArg(0, d_a);
    kernel.setArg(1, d_b);
    kernel.setArg(2, d_c);
    kernel.setArg(3, MATRIX_WIDTH);

    std::cout << "Starting calc\n";
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(MATRIX_WIDTH), cl::NullRange);

    std::cout << "Starting memory copy back\n";

    try {
        queue.enqueueReadBuffer(d_c, CL_TRUE, 0, sizeof(float) * MATRIX_LENGTH, output);
    }
    catch (cl::Error& e) {
        std::cout << "Error: " << e.what() << " " << e.err() << "\n";
    }

    // std::cout << "Stress test\n";

    // while(1) {
    //     std::cout << "Starting calc\n";
    //     queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(MATRIX_WIDTH), cl::NullRange);

    //     std::cout << "Starting memory copy back\n";

    //     try {
    //         queue.enqueueReadBuffer(d_c, CL_TRUE, 0, sizeof(float) * MATRIX_LENGTH, output);
    //     }
    //     catch (cl::Error& e) {
    //         std::cout << "Error: " << e.what() << " " << e.err() << "\n";
    //     }
    // }

    // cl::copy(queue, d_c, output, output + (MATRIX_LENGTH));
    queue.finish();

    std::cout << "Finished\n";

    // std::cout << "Result:\n";
    // for (int i = 0; i < h_c.size(); i++) {
    //     std::cout << h_c[i] << " ";
    // }

    std::cout << "\n";
}