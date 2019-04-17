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

    // setup functions
    auto f = "ret = v > 0 ? v : v * 0.1f;";
    auto df = "ret = v > 0 ? 1 : 0.1f;";
    auto dcost = "2 * v;";

    // setup core generator
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
    ncf::Stock<float> il_stock(il, 1);
    ncf::Stock<float> hl_stock(hl, 1);
    ncf::Stock<float> ol_stock(ol, 1);

    video << il_stock << hl_stock << ol_stock;

    // query
    il.query(data, il_stock, video);
    hl.query(il_stock, hl_stock, video);
    ol.query(hl_stock, ol_stock, video);

    // error
    ol.error(answer, ol_stock, video);
    hl.error(ol_stock, hl_stock, video);

    video >> il_stock >> hl_stock >> ol_stock;

    // output
    std::cout << "Data" << std::endl;
    std::cout << data << std::endl;
    
    std::cout << "Net" << std::endl;
    std::cout << il_stock.getConstOut() << std::endl;
    std::cout << hl_stock.getConstOut() << std::endl;
    std::cout << ol_stock.getConstOut() << std::endl;

    std::cout << "Answer" << std::endl;
    std::cout << answer << std::endl;

    std::cout << "Error" << std::endl;
    std::cout << hl_stock.getConstError() << std::endl;
    std::cout << ol_stock.getConstError() << std::endl;

    std::cout << "Total error " << ol.cost(ol_stock, ncf::cost::mse<float>) << std::endl;

    ecl::System::release();
    return 0;
}