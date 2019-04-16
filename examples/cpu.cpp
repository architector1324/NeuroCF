#include <iostream>
#include <NeuroCF/NeuroCF.hpp>

int main()
{
    mcf::Mat<float> data(5, 1);
    mcf::Mat<float> answer(3, 1);

    data.full(2.0f);
    answer.full(3.0f);

    auto f = [](const float& v){
        return v > 0 ? v : v * 0.1f;
    };
    auto df = [](const float& v){
        return v > 0 ? 1 : 0.1f;
    };

    auto cost = [](const float& v){
        return v * v;
    };
    auto dcost = [](const float& v){
        return 2 * v;
    };

    auto coregen = [](mcf::Mat<float>& A){
        A.gen([](size_t i, size_t j){
            return (float)(i + j) / 10.0f;
        });
    };

    // setup net
    ncf::Layer<float> il(5);
    il.setActivation(f);

    ncf::Layer<float> hl(2);
    hl.setActivation(f);
    hl.setDerivative(df);
    hl.setCoreGen(coregen);

    ncf::Layer<float> ol(3);
    ol.setActivation(f);
    ol.setCoreGen(coregen);

    // setup containers
    mcf::Mat<float> hl_preout(2, 1);

    mcf::Mat<float> il_out(5, 1);
    mcf::Mat<float> hl_out(2, 1);
    mcf::Mat<float> ol_out(3, 1);

    mcf::Mat<float> ol_error(3, 1);
    mcf::Mat<float> hl_error(2, 1);

    mcf::Mat<float> ol_grad(3, 2);
    mcf::Mat<float> hl_grad(2, 5);

    // query
    il.query(data, il_out);
    hl.query(il_out, hl_preout, hl_out, il);
    ol.query(hl_out, ol_out, ol_out, hl);

    // error
    ol.error(answer, ol_out, ol_error);
    hl.error(ol_error, hl_preout, hl_error, ol);

    float e = ol.cost(ol_error, cost);

    // grad
    ol.grad(ol_error, hl_out, ol_grad, dcost);
    hl.grad(hl_error, il_out, hl_grad, dcost);

    //output
    std::cout << "Data" << std::endl;
    std::cout << data << std::endl;

    std::cout << "Net" << std::endl;
    std::cout << il_out << std::endl;
    std::cout << hl_out << std::endl;
    std::cout << ol_out << std::endl;

    std::cout << "Answer" << std::endl;
    std::cout << answer << std::endl;

    std::cout << "Total error = " << e << std::endl;

    std::cout << "Grad" << std::endl;
    std::cout << hl_grad << std::endl;
    std::cout << ol_grad;

    return 0;
}