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
	// setup computer
	auto plat = ecl::System::getPlatform(0);
	ecl::Computer video(0, plat, ecl::DEVICE::GPU);

	// setup data
	mcf::Mat<float> data(500, 100);
	mcf::Mat<float> answer(300, 100);

	data.full(2.0f);
	answer.full(3.0f);

	video << data << answer;

	// setup functions
	auto lrelu = "ret = v > 0 ? v : v * 0.1f;";
	auto div_lrelu = "ret = v > 0 ? 1 : 0.1f;";
	auto div_mse = "ret = 2 * v;";

	// setup core generator
	auto coregen = [](mcf::Mat<float>& A, ecl::Computer& video) {
		A.full(0.01f, video);
	};

	// setup net
	ncf::Net<float> net({ 500, 200, 300 });
	net.setActivations(lrelu);
	net.setDerivatives({ 1 }, div_lrelu);
	net.setCoreGens({ 1, 2 }, coregen);

	video << net;

	// setup containers
	ncf::StockPool<float> pool(net, 100);

	video << pool;

	// fit
	ncf::FitFrame<float> frame = { data, answer, pool, ncf::cost::mse<float>, div_mse };

	float e = 1.0f;
	executionTime([&] {
		e = net.fit(frame, 0.025f, 5, 0.001f, video);
	}, 5);

	std::cout << "Total error " << e << std::endl;

	ecl::System::release();
	return 0;
}