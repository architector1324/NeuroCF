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

	// setup layers
	ncf::Layer<float> il(5);
	il.setActivation(lrelu);

	ncf::Layer<float> hl(2);
	hl.setActivation(lrelu);
	hl.setDerivative(div_lrelu);
	hl.setCoreGen(coregen);

	ncf::Layer<float> ol(3);
	ol.setActivation(lrelu);
	ol.setCoreGen(coregen);

	// setup matrices stocks
	ncf::Stock<float> il_stock(il, 1);
	ncf::Stock<float> hl_stock(hl, 1);
	ncf::Stock<float> ol_stock(ol, 1);

	video << il_stock << hl_stock << ol_stock;

	// fit
	float e = 1.0f;

	for (size_t i = 0; i < 100; i++) {
		// query
		il.query(data, il_stock, video);
		hl.query(il_stock, hl_stock, video);
		ol.query(hl_stock, ol_stock, video);

		// error
		ol.error(answer, ol_stock, video);
		hl.error(ol_stock, hl_stock, video);

		// video >> ol_stock.getError(); // this is better to use
		video >> ol_stock;
		e = ol.cost(ol_stock, ncf::cost::mse<float>);
		if (e < 0.001f) break;

		// grad
		hl.grad(il_stock, hl_stock, div_mse, video);
		ol.grad(hl_stock, ol_stock, div_mse, video);

		// train
		hl.train(il_stock, hl_stock, 0.025f, video);
		ol.train(hl_stock, ol_stock, 0.025f, video);
	}

	video >> ol_stock;

	// output
	std::cout << "Data" << std::endl;
	std::cout << data << std::endl;

	std::cout << "Answer" << std::endl;
	std::cout << answer << std::endl;

	std::cout << "Output:" << std::endl;
	std::cout << ol_stock << std::endl;

	std::cout << "Total error " << e << std::endl;

	ecl::System::release();
	return 0;
}