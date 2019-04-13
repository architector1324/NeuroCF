#include <iostream>
#include <NeuroCF/NeuroCF.hpp>

int main()
{
    mcf::Mat<float> data(5, 1);
    data.full(2.0f);

    auto f = [](const float& v){
        return v > 0 ? v : v * 0.1f;
    };
    auto coregen = [](mcf::Mat<float>& A){
        A.gen([](size_t i, size_t j){
            return (float)(i + j) / 10.0f;
        });
    };

    // setup net
    ncf::Layer<float> il(5);
    il.setActivation(f);

    ncf::Layer<float> ol(3);
    ol.setActivation(f);
    ol.setCoreGen(coregen);

    // setup containers
    mcf::Mat<float> il_out(5, 1);
    mcf::Mat<float> ol_out(3, 1);

    // query
    il.query(data, il_out);
    ol.query(il_out, ol_out, ol_out, il);

    //output
    std::cout << data << std::endl;
    std::cout << il_out << std::endl;
    std::cout << ol_out << std::endl;

    return 0;
}