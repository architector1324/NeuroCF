#include <iostream>
#include <NeuroCF/NeuroCF.hpp>

int main()
{
    mcf::Mat<float> data(5, 1);
    data.full(-2.0f);

    auto f = "ret = v > 0 ? v : v * 0.1f;";

    // setup net
    ncf::Layer<float> il(5);
    il.setActivation(f);

    // setup containers
    mcf::Mat<float> il_out(5, 1);

    // setup gpu
    auto p = ecl::System::getPlatform(0);
    ecl::Computer video(0, p, ecl::DEVICE::GPU);

    video << data << il_out;

    // query
    il.query(data, il_out, video);
    video >> il_out;

    //output
    std::cout << data << std::endl;
    std::cout << il_out;

    // free
    data.release(video);
    il_out.release(video);

    ecl::System::free();

    return 0;
}