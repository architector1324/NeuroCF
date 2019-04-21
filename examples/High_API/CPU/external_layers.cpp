#include <iostream>
#include <NeuroCF/NeuroCF.hpp>

int main()
{
	// setup data
	mcf::Mat<float> data0(5, 1);
	mcf::Mat<float> data1(3, 1);

	data0.full(1.0f);
	data1.full(2.0f);

	mcf::Mat<float> answer0(3, 1);
	mcf::Mat<float> answer1(2, 1);

	answer0.full(1.0f);
	answer1.full(2.0f);

	// setup core generator
	auto coregen = [](mcf::Mat<float>& A) {
		A.full(0.01f);
	};

	// setup layers
	ncf::Layer<float> h0(5);
	h0.setActivation(ncf::activation::lrelu<float>);
	h0.setDerivative(ncf::derivative::activation::lrelu<float>);
	h0.setCoreGen(coregen);

	ncf::Layer<float> h1(2);
	h1.setActivation(ncf::activation::lrelu<float>);
	h1.setDerivative(ncf::derivative::activation::lrelu<float>);
	h1.setCoreGen(coregen);

	ncf::Layer<float> h2(3);
	h2.setActivation(ncf::activation::lrelu<float>);
	h2.setDerivative(ncf::derivative::activation::lrelu<float>);
	h2.setCoreGen(coregen);

	// setup net0
	ncf::Net<float> net0;

	net0.push_back(&h0);
	net0.push_back(&h1);
	net0.push_back(&h2);

	// setup net1
	ncf::Net<float> net1({&h2, &h0, &h1});

	// setup matrices containers
	ncf::StockPool<float> pool0(net0, 1);
	ncf::StockPool<float> pool1(net1, 1);

	// fit
	ncf::FitFrame<float> frame0 = {data0, answer0, pool0, ncf::cost::mse<float>, ncf::derivative::cost::mse<float>};
	ncf::FitFrame<float> frame1 = {data1, answer1, pool1, ncf::cost::mse<float>, ncf::derivative::cost::mse<float>};

	net0.fit(frame0, 0.025f, 100, 0.001f);
	net1.fit(frame1, 0.025f, 100, 0.001f);

	// output0
	std::cout << "Data0" << std::endl;
	std::cout << data0 << std::endl;

	std::cout << "Answer0" << std::endl;
	std::cout << answer0 << std::endl;

	std::cout << "Net0" << std::endl;
	std::cout << pool0 << std::endl;

	// output1
	std::cout << "Data1" << std::endl;
	std::cout << data1 << std::endl;

	std::cout << "Answer1" << std::endl;
	std::cout << answer1 << std::endl;

	std::cout << "Net1" << std::endl;
	std::cout << pool1 << std::endl;

	return 0;
}