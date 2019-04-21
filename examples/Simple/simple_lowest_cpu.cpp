#include <iostream>
#include <NeuroCF/NeuroCF.hpp>

int main()
{
	// setup data
	mcf::Mat<float> data(5, 1);
	mcf::Mat<float> answer(3, 1);

	data.full(2.0f);
	answer.full(3.0f);

	// setup core generator
	auto coregen = [](mcf::Mat<float>& A) {
		A.full(0.01f);
	};

	// setup layers
	ncf::Layer<float> il(5);
	il.setActivation(ncf::activation::lrelu<float>);

	ncf::Layer<float> hl(2);
	hl.setActivation(ncf::activation::lrelu<float>);
	hl.setDerivative(ncf::derivative::activation::lrelu<float>);
	hl.setCoreGen(coregen);

	ncf::Layer<float> ol(3);
	ol.setActivation(ncf::activation::lrelu<float>);
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

	// fit
	float e = 1.0f;

	for (size_t i = 0; i < 100; i++) {
		// query
		il.query(data, il_out);
		hl.query(il_out, hl_preout, hl_out, il);
		ol.query(hl_out, ol_out, ol_out, hl);

		// error
		ol.error(answer, ol_out, ol_error);
		hl.error(ol_error, hl_preout, hl_error, ol);

		e = ol.cost(ol_error, ncf::cost::mse<float>);
		if (e < 0.001f) break;

		// grad
		hl.grad(hl_error, il_out, hl_grad, ncf::derivative::cost::mse<float>);
		ol.grad(ol_error, hl_out, ol_grad, ncf::derivative::cost::mse<float>);

		// train
		hl.train(hl_grad, il, 0.025f);
		ol.train(ol_grad, hl, 0.025f);
	}

	// output
	std::cout << "Data" << std::endl;
	std::cout << data << std::endl;

	std::cout << "Answer" << std::endl;
	std::cout << answer << std::endl;

	std::cout << "Output:" << std::endl;
	std::cout << ol_out << std::endl;

	std::cout << "Total error " << e << std::endl;

	return 0;
}