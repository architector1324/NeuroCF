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
    auto coregen = [](mcf::Mat<float>& A){
        A.gen([](size_t i, size_t j){
            return (float)(i + j) / 10.0f;
        });
    };

    // setup net
    ncf::Layer<float> il(5);
    il.setActivation(ncf::activation::lrelu<float>);

    ncf::Layer<float> hl(2);
    hl.setActivation(ncf::activation::lrelu<float>);
    hl.setDerivative(ncf::derivative::activation::lrelu<float>);
    hl.setCoreGen(coregen);

    ncf::Layer<float> ol(3);
    ol.setActivation(ncf::activation::lrelu<float>);
    ol.setCoreGen(coregen);

    // setup containers
    ncf::Stock<float> il_stock(il, 1);
    ncf::Stock<float> hl_stock(hl, 1);
    ncf::Stock<float> ol_stock(ol, 1);

	// fit
	float e = 1.0f;

	for (size_t i = 0; i < 100; i++) {
		// query
		il.query(data, il_stock);
		hl.query(il_stock, hl_stock);
		ol.query(hl_stock, ol_stock);

		// error
		ol.error(answer, ol_stock);
		hl.error(ol_stock, hl_stock);

		e = ol.cost(ol_stock, ncf::cost::mse<float>);
		std::cout << "Total error " << e << std::endl;

		if (e < 0.001f) break;

		// grad
		hl.grad(il_stock, hl_stock, ncf::derivative::cost::mse<float>);
		ol.grad(hl_stock, ol_stock, ncf::derivative::cost::mse<float>);

		// train
		hl.train(il_stock, hl_stock, 0.025);
		ol.train(hl_stock, ol_stock, 0.025);
	}
	std::cout << "Total error " << e << std::endl;

    // output
    std::cout << "Data" << std::endl;
    std::cout << data << std::endl;
    
    std::cout << "Net" << std::endl;
    std::cout << il_stock.getConstOut() << std::endl;
    std::cout << hl_stock.getConstOut() << std::endl;
    std::cout << ol_stock.getConstOut() << std::endl;

    std::cout << "Answer" << std::endl;
    std::cout << answer << std::endl;

    return 0;
}