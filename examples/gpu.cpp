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

    auto f = "ret = v > 0 ? v : v * 0.1f;";
    auto df = "ret = v > 0 ? 1 : 0.1f;";
    auto dcost = "2 * v;";

    auto coregen = [](mcf::Mat<float>& A){
        A.gen([](size_t i, size_t j){
            return (float)(i + j) / 10.0f;
        });
    };

    video << data << answer;

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

    video << il << hl << ol;

    // setup containers
    mcf::Mat<float> hl_preout(2, 1);

    mcf::Mat<float> il_out(5, 1);
    mcf::Mat<float> hl_out(2, 1);
    mcf::Mat<float> ol_out(3, 1);

    mcf::Mat<float> ol_error(3, 1);
    mcf::Mat<float> hl_error(2, 1);

    mcf::Mat<float> ol_grad(3, 2);
    mcf::Mat<float> hl_grad(2, 5);

    video << hl_preout;
    video << il_out << hl_out << ol_out;
    video << ol_error << hl_error;
    video << ol_grad << hl_grad;

    // query
    il.query(data, il_out, video);
    hl.query(il_out, hl_preout, hl_out, il, video);
    ol.query(hl_out, ol_out, ol_out, hl, video);

    // fit
    float e = 1.0f;

    for(size_t i = 0; i < 100; i++){
        // query
        il.query(data, il_out, video);
        hl.query(il_out, hl_preout, hl_out, il, video);
        ol.query(hl_out, ol_out, ol_out, hl, video);

        // error
        ol.error(answer, ol_out, ol_error, video);
        hl.error(ol_error, hl_preout, hl_error, ol, video);

        video >> ol_error;
        e = ol.cost(ol_error, ncf::cost::mse<float>);
        if(e < 0.001f) break;

        // grad
        ol.grad(ol_error, hl_out, ol_grad, dcost, video);
        hl.grad(hl_error, il_out, hl_grad, dcost, video);

        // train
        ncf::optimizer::gd<float>(ol.getCore(2), ol_grad, 0.025, video);
        ncf::optimizer::gd<float>(hl.getCore(5), hl_grad, 0.025, video);

        std::cout << "Total error = " << e << std::endl;
    }
    std::cout << "Total error = " << e << std::endl;

    video >> il_out >> hl_out >> ol_out;
    //output
    std::cout << "Data" << std::endl;
    std::cout << data << std::endl;

    std::cout << "Net" << std::endl;
    std::cout << il_out << std::endl;
    std::cout << hl_out << std::endl;
    std::cout << ol_out << std::endl;

    std::cout << "Answer" << std::endl;
    std::cout << answer << std::endl;

    return 0;
}