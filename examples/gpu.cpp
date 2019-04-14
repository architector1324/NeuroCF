#include <iostream>
#include <NeuroCF/NeuroCF.hpp>

int main()
{
    mcf::Mat<float> data(5, 1);
    data.full(2.0f);

    auto f = "ret = v > 0 ? v : v * 0.1f;";
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

    // setup gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << data << ol;
    video << il_out << ol_out;

    // query
    il.query(data, il_out, video);
    ol.query(il_out, ol_out, ol_out, il, video);

    video >> il_out >> ol_out;

    //output
    std::cout << data << std::endl;
    std::cout << il_out << std::endl;
    std::cout << ol_out << std::endl;

    ecl::System::release();

    return 0;
}