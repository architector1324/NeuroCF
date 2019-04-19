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
    net.setDerivatives(ncf::derivative::activation::lrelu<float>);
    net.setCoreGens(coregen);

    // setup containers
    ncf::StockPool<float> pool(net, 1);

    return 0;
}