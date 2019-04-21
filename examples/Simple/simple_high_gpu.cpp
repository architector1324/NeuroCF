#include <iostream>
#include <NeuroCF/NeuroCF.hpp>

int main()
{
	// setup computer
	auto plat = ecl::System::getPlatform(0);
	ecl::Computer video(0, plat, ecl::DEVICE::GPU);

	// setup data
	mcf::Mat<float> data(5, 1);
	mcf::Mat<float> answer(3, 1);

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
	ncf::Net<float> net({ 5, 2, 3 });
	net.setActivations(lrelu);
	net.setDerivatives({ 1 }, div_lrelu);
	net.setCoreGens({ 1, 2 }, coregen);

	// setup matrices stocks pool
	ncf::StockPool<float> pool(net, 1);
	video << pool;

	// fit
	float e = 1.0f;

	for (size_t i = 0; i < 100; i++) {
		net.query(data, pool, video); // query
		net.error(answer, pool, video); // error

		// video >> pool.getLastStock().getError(); // is better to use
		video >> pool;
		e = net.cost(pool, ncf::cost::mse<float>);
		if (e < 0.001f) break;

		net.grad(pool, div_mse, video); // grad
		net.train(pool, 0.025f, video); // train
	}

	video >> pool;

	// output
	std::cout << "Data" << std::endl;
	std::cout << data << std::endl;

	std::cout << "Answer" << std::endl;
	std::cout << answer << std::endl;

	std::cout << "Output:" << std::endl;
	std::cout << pool << std::endl;

	std::cout << "Total error " << e << std::endl;

	ecl::System::release();
	return 0;
}