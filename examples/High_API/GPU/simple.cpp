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

	video << data << answer;

    // setup functions
    auto lrelu = "ret = v > 0 ? v : v * 0.1f;";
    auto div_lrelu = "ret = v > 0 ? 1 : 0.1f;";
    auto div_mse = "ret = 2 * v;";

    // setup core generator
    auto coregen = [](mcf::Mat<float>& A, ecl::Computer& video){
		A.full(0.01f, video);
    };

    // setup net
    ncf::Net<float> net({5, 2, 3});
    net.setActivations(lrelu);
    net.setDerivatives({1}, div_lrelu);
    net.setCoreGens({1, 2}, coregen);

    video << net;

    // setup containers
    ncf::StockPool<float> pool(net, 1);
	
	video << pool;

	// fit
	ncf::FitFrame<float> frame = {data, answer, pool, ncf::cost::mse<float>, div_mse};

	float e = net.fit(frame, 0.025f, 100, 0.001f, video);
	video >> pool;

    // output
	std::cout << "Data" << std::endl;
	std::cout << data << std::endl;

	std::cout << "Answer" << std::endl;
	std::cout << answer << std::endl;

	std::cout << "Output:" << std::endl;
	std::cout << pool << std::endl;

	std::cout << "Total error " << e << std::endl;

    ecl::System::release();
    return 0;
}