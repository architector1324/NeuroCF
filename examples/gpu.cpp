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
    auto dcost = "ret = 2 * v;";

    // setup core generator
    auto coregen = [](mcf::Mat<float>& A, ecl::Computer& video){
        A.gen("ret = (float)(i + j) / 10.0f;", video);
    };

    video << data << answer;

    // setup net
    ncf::Net<float> net({5, 2, 3});
    net.setActivations(f);
    net.setDerivatives({1}, df);
    net.setCoreGens({1, 2}, coregen);

    video << net;

    // setup containers
    ncf::StockPool<float> pool(net, 1);

    video << pool;

    // compute
    net.query(data, pool, video);
    net.error(answer, pool, video);

    video >> pool;

    // output
    std::cout << "Data" << std::endl;
    std::cout << data << std::endl;

    std::cout << "Net" << std::endl;
    for(size_t i = 0; i < pool.getStocksCount(); i++)
        std::cout << pool.getConstStock(i).getConstOut() << std::endl;

    std::cout << "Answer" << std::endl;
    std::cout << answer << std::endl;

    std::cout << "Error" << std::endl;
    for(size_t i = 1; i < pool.getStocksCount(); i++)
        std::cout << pool.getConstStock(i).getConstError() << std::endl;

    ecl::System::release();
    return 0;
}