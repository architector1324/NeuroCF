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
    ncf::Net<float> net({5, 2, 3});
    net.setActivations(ncf::activation::lrelu<float>);
    net.setDerivatives({1}, ncf::derivative::activation::lrelu<float>);
    net.setCoreGens({1, 2}, coregen);

    // setup containers
    ncf::StockPool<float> pool(net, 1);

    // query
    net.query(data, pool);

    // output
    for(size_t i = 0; i < pool.getStocksCount(); i++)
        std::cout << pool.getConstStock(i).getConstOut() << std::endl;

    return 0;
}