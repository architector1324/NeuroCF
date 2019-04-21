#include <iostream>
#include <chrono>
#include <NeuroCF/NeuroCF.hpp>

void executionTime(const std::function<void()>& f, size_t times = 1) {
	for (size_t i = 0; i < times; i++) {
		auto start = std::chrono::high_resolution_clock::now();
		f();
		auto end = std::chrono::high_resolution_clock::now();

		auto mcs = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << mcs.count() << " mcs (" << ms.count() << " ms)" << std::endl;
	}
}

int main()
{
    // setup data
    mcf::Mat<float> data(500, 100);
    mcf::Mat<float> answer(300, 100);

    data.full(2.0f);
    answer.full(3.0f);

    // setup core generator
    auto coregen = [](mcf::Mat<float>& A){
		A.full(0.01f);
    };

    // setup net
    ncf::Net<float> net({500, 200, 300});
    net.setActivations(ncf::activation::lrelu<float>);
    net.setDerivatives({1}, ncf::derivative::activation::lrelu<float>);
    net.setCoreGens({1, 2}, coregen);

    // setup matrices containers
    ncf::StockPool<float> pool(net, 100);

    // fit
	ncf::FitFrame<float> frame = {data, answer, pool, ncf::cost::mse<float>, ncf::derivative::cost::mse<float>};

	float e = 1.0f;

	executionTime([&] {
		e = net.fit(frame, 0.025f, 5, 0.001f);
	}, 5);
	
	std::cout << "Total error " << e << std::endl;

    return 0;
}