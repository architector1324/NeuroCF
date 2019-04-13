#include <iostream>
#include <NeuroCF/NeuroCF.hpp>

int main()
{
    mcf::Mat<float> data(5, 1);
    data.full(-2.0f);

    auto f = [](const float& v){
        return v > 0 ? v : v * 0.1f;
    };

    // setup net
    ncf::Layer<float> il(5);
    il.setActivation(f);

    // setup containers
    mcf::Mat<float> il_out(5, 1);

    // query
    il.query(data, il_out);

    //output
    std::cout << data << std::endl;
    std::cout << il_out;

    return 0;
}