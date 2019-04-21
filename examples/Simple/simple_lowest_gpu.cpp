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

	// setup matrices
	mcf::Mat<float> il_out(5, 1);

	mcf::Mat<float> hl_preout(2, 1);
	mcf::Mat<float> hl_out(2, 1);
	mcf::Mat<float> hl_error(2, 1);
	mcf::Mat<float> hl_grad(2, 5);

	mcf::Mat<float> ol_out(3, 1);
	mcf::Mat<float> ol_error(3, 1);
	mcf::Mat<float> ol_grad(3, 2);

	video << il_out;
	video << hl_preout << hl_out << hl_error << hl_grad;
	video << ol_out << ol_error << ol_grad;

	// fit
	float e = 1.0f;

	for (size_t i = 0; i < 100; i++) {
		// query
		il.query(data, il_out, video);
		hl.query(il_out, hl_preout, hl_out, il, video);
		ol.query(hl_out, ol_out, ol_out, hl, video);

		// error
		ol.error(answer, ol_out, ol_error, video);
		hl.error(ol_error, hl_preout, hl_error, ol, video);

		video >> ol_error;
		e = ol.cost(ol_error, ncf::cost::mse<float>);
		if (e < 0.001f) break;

		// grad
		hl.grad(hl_error, il_out, hl_grad, div_mse, video);
		ol.grad(ol_error, hl_out, ol_grad, div_mse, video);

		// train
		hl.train(hl_grad, il, 0.025f, video);
		ol.train(ol_grad, hl, 0.025f, video);
	}

	video >> ol_out;

	// output
	std::cout << "Data" << std::endl;
	std::cout << data << std::endl;

	std::cout << "Answer" << std::endl;
	std::cout << answer << std::endl;

	std::cout << "Output:" << std::endl;
	std::cout << ol_out << std::endl;

	std::cout << "Total error " << e << std::endl;

	ecl::System::release();
	return 0;
}