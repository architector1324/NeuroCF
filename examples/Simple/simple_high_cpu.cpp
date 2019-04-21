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

	// setup net
	ncf::Net<float> net({ 5, 2, 3 });
	net.setActivations(ncf::activation::lrelu<float>);
	net.setDerivatives({ 1 }, ncf::derivative::activation::lrelu<float>);
	net.setCoreGens({ 1, 2 }, coregen);

	// setup matrices stocks pool
	ncf::StockPool<float> pool(net, 1);

	// fit
	float e = 1.0f;

	for (size_t i = 0; i < 100; i++) {
		net.query(data, pool); // query
		net.error(answer, pool); // error

		e = net.cost(pool, ncf::cost::mse<float>);
		if (e < 0.001f) break;
		
		net.grad(pool, ncf::derivative::cost::mse<float>); // grad
		net.train(pool, 0.025f); // train
	}

	// output
	std::cout << "Data" << std::endl;
	std::cout << data << std::endl;

	std::cout << "Answer" << std::endl;
	std::cout << answer << std::endl;

	std::cout << "Output:" << std::endl;
	std::cout << pool << std::endl;

	std::cout << "Total error " << e << std::endl;

	return 0;
}