#pragma once

#include "MatrixCF.hpp"

namespace ncf{
    using namespace mcf;
    using namespace ecl;

    // Low-level API
    template<typename T>
    class Stock;

    template<typename T>
    class Layer{
    private:
        std::map<std::size_t, Mat<T>> core;
        std::size_t neurons = 0;

        std::function<T(const T&)> activation = nullptr;
        std::function<T(const T&)> derivative = nullptr;

        std::string computer_activation = "";
        std::string computer_derivative = "";

        std::function<void(Mat<T>&)> coregen = nullptr;
		std::function<void(Mat<T>&, Computer&)> computer_coregen = nullptr;

    public:
        Layer() = delete;
        explicit Layer(std::size_t);

        void send(Computer&);
        void receive(Computer&);
        void grab(Computer&);
        void release(Computer&);

        template<typename U>
        friend Computer& operator<<(Computer&, Layer<U>&);
        template<typename U>
        friend Computer& operator>>(Computer&, Layer<U>&);

        bool checkCore(std::size_t) const;
        void createCore(std::size_t);
		void createCore(std::size_t, Computer&);
        void releaseCore(std::size_t);

        void setActivation(const std::function<T(const T&)>&);
        void setDerivative(const std::function<T(const T&)>&);

        void setActivation(const std::string&);
        void setDerivative(const std::string&);

        void setCoreGen(const std::function<void(Mat<T>&)>&);
		void setCoreGen(const std::function<void(Mat<T>&, Computer&)>&);

        std::size_t getNeurons() const;
        Mat<T>& getCore(std::size_t);
        const Mat<T>& getConstCore(std::size_t) const;
        const std::function<T(const T&)>& getActivation() const;
        const std::function<T(const T&)>& getDerivative() const;
        const std::string& getComputerActivation() const;
        const std::string& getComputerDerivative() const;
        const std::function<void(Mat<T>&)>& getCoreGen() const;
		const std::function<void(Mat<T>&, Computer&)>& getComputerCoreGen() const;

        // Low-level methods
        void query(const Mat<T>&, Mat<T>&) const;
        void query(const Mat<T>&, Mat<T>&, Computer&) const;

        void query(const Mat<T>&, Mat<T>&, Mat<T>&, const Layer<T>&);
        void query(const Mat<T>&, Mat<T>&, Mat<T>&, const Layer<T>&, Computer&);

        void error(const Mat<T>&, const Mat<T>&, Mat<T>&) const;
        void error(const Mat<T>&, const Mat<T>&, Mat<T>&, Computer&) const;

        void error(const Mat<T>&, Mat<T>&, Mat<T>&, const Layer<T>&) const;
        void error(const Mat<T>&, Mat<T>&, Mat<T>&, const Layer<T>&, Computer&) const;

        T cost(const Mat<T>&, const std::function<T(const T&)>&) const;

        void grad(Mat<T>&, const Mat<T>&, Mat<T>&, const std::function<T(const T&)>&) const;
        void grad(Mat<T>&, const Mat<T>&, Mat<T>&, const std::string&, Computer&) const;

        void train(Mat<T>&, const Layer<T>&, T);
        void train(Mat<T>&, const Layer<T>&, T, Computer& video);

        // High-level methods
        void query(const Mat<T>&, Stock<T>&) const;
        void query(const Mat<T>&, Stock<T>&, Computer&) const;

        void query(const Stock<T>&, Stock<T>&);
        void query(const Stock<T>&, Stock<T>&, Computer&);

        void error(const Mat<T>&, Stock<T>&) const;
        void error(const Mat<T>&, Stock<T>&, Computer&) const;

        void error(const Stock<T>&, Stock<T>&) const;
        void error(const Stock<T>&, Stock<T>&, Computer&) const;

        T cost(const Stock<T>&, const std::function<T(const T&)>&) const;

		void grad(const Stock<T>&, Stock<T>&, const std::function<T(const T&)>&) const;
		void grad(const Stock<T>&, Stock<T>&, const std::string&, Computer&) const;

		void train(const Stock<T>&, Stock<T>&, T);
		void train(const Stock<T>&, Stock<T>&, T, Computer&);
    };

    template<typename T>
    class Stock{
    private:
        std::map<std::size_t, Mat<T>> grad;
        Mat<T> preout;
        Mat<T> out;
        Mat<T> error;

        const Layer<T>& layer;
    public:
        Stock(const Layer<T>&, std::size_t);

        void send(Computer&);
        void receive(Computer&);
        void grab(Computer&);
        void release(Computer&);

        template<typename U>
        friend Computer& operator<<(Computer&, Stock<U>&);
        template<typename U>
        friend Computer& operator>>(Computer&, Stock<U>&);

        Mat<T>& getPreout();
        Mat<T>& getOut();
        Mat<T>& getError();
        Mat<T>& getGrad(std::size_t);

        const Mat<T>& getConstPreout() const;
        const Mat<T>& getConstOut() const;
        const Mat<T>& getConstError() const;
        const Layer<T>& getLayer() const;

        bool checkGrad(std::size_t) const;
        void createGrad(std::size_t);
		void createGrad(std::size_t, Computer&);
        void releaseGrad(std::size_t);
    };

    // High-level API
    template<typename T>
    class StockPool;

    template<typename T>
    class Net{
    private:
        std::vector<std::pair<Layer<T>*, bool>> layers;
    public:
        Net();
        explicit Net(const std::vector<std::size_t>&);

        void send(Computer&);
        void receive(Computer&);
        void grab(Computer&);
        void release(Computer&);

        template<typename U>
        friend Computer& operator<<(Computer&, Net<U>&);
        template<typename U>
        friend Computer& operator>>(Computer&, Net<U>&);

        // total setters
        void setActivations(const std::function<T(const T&)>&);
        void setDerivatives(const std::function<T(const T&)>&);

        void setActivations(const std::string&);
        void setDerivatives(const std::string&);

        void setCoreGens(const std::function<void(Mat<T>&)>&);
		void setCoreGens(const std::function<void(Mat<T>&, Computer&)>&);

        // partial setters

        std::size_t getLayersCount() const;
        const Layer<T>* getConstLayer(std::size_t) const;
        Layer<T>* getLayer(std::size_t);

        ~Net();
    };

    template<typename T>
    class StockPool{
    private:
        std::vector<std::pair<Stock<T>*, bool>> stocks;
    public:
        StockPool();
        StockPool(const Net<T>&, std::size_t);

        void send(Computer&);
        void receive(Computer&);
        void grab(Computer&);
        void release(Computer&);

        template<typename U>
        friend Computer& operator<<(Computer&, Net<U>&);
        template<typename U>
        friend Computer& operator>>(Computer&, Net<U>&);

        std::size_t getStocksCount() const;
        const Stock<T>* getConstStock(std::size_t) const;
        Stock<T>* getStock(std::size_t);

        ~StockPool();
    };
}

namespace ncf{
    namespace activation {
		template<typename T>
		T relu(const T& v){
			return v > 0 ? v : 0;
		}

		template<typename T>
		T lrelu(const T& v){
			return v > 0 ? v : v * T(0.1);
		}
	}
	namespace cost {
		template<typename T>
		T mse(const T& v) {
			return v * v;
		}
	}

	namespace derivative {
		namespace activation {
			template<typename T>
			T relu(const T& v) {
				return v > 0 ? 1 : 0;
			}

			template<typename T>
			T lrelu(const T& v) {
				return v > 0 ? 1 : T(0.1);
			}
		}
		namespace cost {
			template<typename T>
			T mse(const T& v) {
				return 2 * v;
			}
		}
	}

    namespace optimizer{
        template<typename T>
        void gd(Mat<T>& X, Mat<T>& grad, T learning_rate){
            grad.mul(learning_rate, grad);
            X.sub(grad, X);
        }
        template<typename T>
        void gd(Mat<T>& X, Mat<T>& grad, T learning_rate, ecl::Computer& video){
            grad.mul(learning_rate, grad, video);
            X.sub(grad, X, video);
        }
    }
}

// IMPLEMENTATION

// Low-level API

// Layer
template<typename T>
ncf::Layer<T>::Layer(std::size_t neurons){
    this->neurons = neurons;
}

template<typename T>
void ncf::Layer<T>::send(ecl::Computer& video){
    for(auto& p : core) video << p.second;
}
template<typename T>
void ncf::Layer<T>::receive(ecl::Computer& video){
    for(auto& p : core) video >> p.second;
}
template<typename T>
void ncf::Layer<T>::grab(ecl::Computer& video){
    for(auto& p : core) p.second.grab(video);
}
template<typename T>
void ncf::Layer<T>::release(ecl::Computer& video){
    for(auto& p : core) p.second.release(video);
}

namespace ncf{
    template<typename T>
    Computer& operator<<(Computer& video, Layer<T>& other){
        other.send(video);
        return video;
    }
    template<typename T>
    Computer& operator>>(Computer& video, Layer<T>& other){
        other.receive(video);
        return video;
    }
}

template<typename T>
bool ncf::Layer<T>::checkCore(std::size_t prev_neurons) const{
    if(core.find(prev_neurons) == core.end()) return false;
    return true;
}
template<typename T>
void ncf::Layer<T>::createCore(std::size_t prev_neurons){
    if(!checkCore(prev_neurons)){
        if(coregen == nullptr)
            throw std::runtime_error("Layer [create core]: coregen method unsetted");

        Mat<T> new_core(neurons, prev_neurons);
        coregen(new_core);
        core.emplace(prev_neurons, std::move(new_core));
    }
}
template<typename T>
void ncf::Layer<T>::createCore(std::size_t prev_neurons, ecl::Computer& video) {
	if (!checkCore(prev_neurons)) {
		Mat<T> new_core(neurons, prev_neurons);
		video << new_core;
		computer_coregen(new_core, video);
		core.emplace(prev_neurons, std::move(new_core));
	}
}
template<typename T>
void ncf::Layer<T>::releaseCore(std::size_t prev_neurons){
    auto it = core.find(prev_neurons);
    if(it != core.end()){
        core.erase(it);
    }
}

template<typename T>
void ncf::Layer<T>::setActivation(const std::function<T(const T&)>& activation){
    this->activation = activation;
}
template<typename T>
void ncf::Layer<T>::setDerivative(const std::function<T(const T&)>& derivative){
    this->derivative = derivative;
}

template<typename T>
void ncf::Layer<T>::setActivation(const std::string& activation){
    this->computer_activation = activation;
}
template<typename T>
void ncf::Layer<T>::setDerivative(const std::string& derivative){
    this->computer_derivative = derivative;
}

template<typename T>
void ncf::Layer<T>::setCoreGen(const std::function<void(mcf::Mat<T>&)>& coregen){
    this->coregen = coregen;
}
template<typename T>
void ncf::Layer<T>::setCoreGen(const std::function<void(mcf::Mat<T>&, ecl::Computer&)>& coregen) {
	this->computer_coregen = coregen;
}

template<typename T>
std::size_t ncf::Layer<T>::getNeurons() const{
    return neurons;
}
template<typename T>
mcf::Mat<T>& ncf::Layer<T>::getCore(std::size_t prev_neurons){
    return core.at(prev_neurons);
}
template<typename T>
const mcf::Mat<T>& ncf::Layer<T>::getConstCore(std::size_t prev_neurons) const{
    return core.at(prev_neurons);
}
template<typename T>
const std::string& ncf::Layer<T>::getComputerActivation() const{
    return computer_activation;
}
template<typename T>
const std::string& ncf::Layer<T>::getComputerDerivative() const{
    return computer_derivative;
}
template<typename T>
const std::function<void(mcf::Mat<T>&)>& ncf::Layer<T>::getCoreGen() const{
    return coregen;
}
template<typename T>
const std::function<void(mcf::Mat<T>&, ecl::Computer&)>& ncf::Layer<T>::getComputerCoreGen() const {
	return computer_coregen;
}
template<typename T>
const std::function<T(const T&)>& ncf::Layer<T>::getActivation() const{
    return activation;
}
template<typename T>
const std::function<T(const T&)>& ncf::Layer<T>::getDerivative() const{
    return derivative;
}

// Low-level methods
template<typename T>
void ncf::Layer<T>::query(const mcf::Mat<T>& in, mcf::Mat<T>& out) const{
    if(activation == nullptr)
        throw std::runtime_error("Layer [query]: activation function unsetted");
    in.map(activation, out);
}
template<typename T>
void ncf::Layer<T>::query(const mcf::Mat<T>& in, mcf::Mat<T>& out, ecl::Computer& video) const{
    in.map(computer_activation, out, video);
}

template<typename T>
void ncf::Layer<T>::query(const mcf::Mat<T>& in, mcf::Mat<T>& preout, mcf::Mat<T>& out, const Layer<T>& prev){
    createCore(prev.neurons);
    
    getCore(prev.neurons).mul(in, preout);
    preout.map(activation, out);
}
template<typename T>
void ncf::Layer<T>::query(const mcf::Mat<T>& in, mcf::Mat<T>& preout, mcf::Mat<T>& out, const Layer<T>& prev, ecl::Computer& video){
    createCore(prev.neurons, video);
    
    getCore(prev.neurons).mul(in, preout, video);
    preout.map(computer_activation, out, video);
}

template<typename T>
void ncf::Layer<T>::error(const mcf::Mat<T>& answer, const mcf::Mat<T>& out, mcf::Mat<T>& error) const{
    answer.sub(out, error);
}
template<typename T>
void ncf::Layer<T>::error(const mcf::Mat<T>& answer, const mcf::Mat<T>& out, mcf::Mat<T>& error, ecl::Computer& video) const{
    answer.sub(out, error, video);
}

template<typename T>
void ncf::Layer<T>::error(const mcf::Mat<T>& next_error, mcf::Mat<T>& preout, mcf::Mat<T>& error, const Layer<T>& next) const{
    if(derivative == nullptr)
        throw std::runtime_error("Layer [query]: derivative function unsetted");

    next.getConstCore(neurons).mul(next_error, error, ncf::TRANSPOSE::FIRST);
    preout.map(derivative, preout);
    error.hadamard(preout, error);
}
template<typename T>
void ncf::Layer<T>::error(const mcf::Mat<T>& next_error, mcf::Mat<T>& preout, mcf::Mat<T>& error, const Layer<T>& next, ecl::Computer& video) const{
    next.getConstCore(neurons).mul(next_error, error, video, ncf::TRANSPOSE::FIRST);
    preout.map(computer_derivative, preout, video);
    error.hadamard(preout, error, video);
}

template<typename T>
T ncf::Layer<T>::cost(const mcf::Mat<T>& error, const std::function<T(const T&)>& cost) const{
    size_t count = error.getH() * error.getW();
    return error.mreduce(cost) / static_cast<T>(count);
}

template<typename T>
void ncf::Layer<T>::grad(mcf::Mat<T>& error, const mcf::Mat<T>& prev_out, mcf::Mat<T>& grad, const std::function<T(const T&)>& div_cost) const{
    size_t count = error.getW() * error.getH();

    error.map(div_cost, error);
    error.mul(prev_out, grad, mcf::TRANSPOSE::SECOND);
    grad.map([&](const T& v){
        return -v / static_cast<T>(count);
    }, grad);
}
template<typename T>
void ncf::Layer<T>::grad(mcf::Mat<T>& error, const mcf::Mat<T>& prev_out, mcf::Mat<T>& grad, const std::string& div_cost, ecl::Computer& video) const{
    std::string count = std::to_string(error.getW() * error.getH());

    error.map(div_cost, error, video);
    error.mul(prev_out, grad, video, mcf::TRANSPOSE::SECOND);
    grad.map("ret = -v / " + count + ";", grad, video);
}

template<typename T>
void ncf::Layer<T>::train(mcf::Mat<T>& grad, const Layer<T>& prev, T learning_rate){
    createCore(prev.neurons);
    optimizer::gd<T>(getCore(prev.neurons), grad, learning_rate);
}
template<typename T>
void ncf::Layer<T>::train(mcf::Mat<T>& grad, const Layer<T>& prev, T learning_rate, ecl::Computer& video){
    createCore(prev.neurons, video);

    optimizer::gd<T>(getCore(prev.neurons), grad, learning_rate, video);
}

// High-level methods
template<typename T>
void ncf::Layer<T>::query(const mcf::Mat<T>& in, Stock<T>& stock) const{
    if(activation == nullptr)
        throw std::runtime_error("Layer [query]: activation function unsetted");
    in.map(activation, stock.getOut());
}
template<typename T>
void ncf::Layer<T>::query(const mcf::Mat<T>& in, Stock<T>& stock, ecl::Computer& video) const{
    in.map(computer_activation, stock.getOut(), video);
}

template<typename T>
void ncf::Layer<T>::query(const Stock<T>& in, Stock<T>& out){
    size_t prev_neurons = in.getLayer().getNeurons();
    createCore(prev_neurons);

    const Mat<T>& in_out = in.getConstOut();

    Mat<T>& out_preout = out.getPreout();
    Mat<T>& out_out = out.getOut();

    getCore(prev_neurons).mul(in_out, out_preout);
    out_preout.map(activation, out_out);
}
template<typename T>
void ncf::Layer<T>::query(const Stock<T>& in, Stock<T>& out, ecl::Computer& video){
    size_t prev_neurons = in.getLayer().getNeurons();
    createCore(prev_neurons, video);
    
    const Mat<T>& in_out = in.getConstOut();

    Mat<T>& out_preout = out.getPreout();
    Mat<T>& out_out = out.getOut();

    getCore(prev_neurons).mul(in_out, out_preout, video);
    out_preout.map(computer_activation, out_out, video);
}

template<typename T>
void ncf::Layer<T>::error(const mcf::Mat<T>& answer, Stock<T>& out) const{
    answer.sub(out.getConstOut(), out.getError());
}
template<typename T>
void ncf::Layer<T>::error(const mcf::Mat<T>& answer, Stock<T>& out, ecl::Computer& video) const{
    answer.sub(out.getConstOut(), out.getError(), video);
}

template<typename T>
void ncf::Layer<T>::error(const Stock<T>& in, Stock<T>& out) const{
    if(derivative == nullptr)
        throw std::runtime_error("Layer [query]: derivative function unsetted");

    const Mat<T>& next_core = in.getLayer().getConstCore(neurons);

    const Mat<T>& in_error = in.getConstError();
    Mat<T>& out_error = out.getError();

    Mat<T>& out_preout = out.getPreout();
    Mat<T>& out_out = out.getOut();

    next_core.mul(in_error, out_error, ncf::TRANSPOSE::FIRST);
    out_preout.map(derivative, out_preout);
    out_error.hadamard(out_preout, out_error);
}
template<typename T>
void ncf::Layer<T>::error(const Stock<T>& in, Stock<T>& out, ecl::Computer& video) const{
    const Mat<T>& next_core = in.getLayer().getConstCore(neurons);

    const Mat<T>& in_error = in.getConstError();
    Mat<T>& out_error = out.getError();

    Mat<T>& out_preout = out.getPreout();
    Mat<T>& out_out = out.getOut();

    next_core.mul(in_error, out_error, video, ncf::TRANSPOSE::FIRST);
    out_preout.map(computer_derivative, out_preout, video);
    out_error.hadamard(out_preout, out_error, video);
}

template<typename T>
T ncf::Layer<T>::cost(const Stock<T>& error, const std::function<T(const T&)>& cost) const{
    const Mat<T>& curr_error = error.getConstError();
    size_t count = curr_error.getH() * curr_error.getW();
    return curr_error.mreduce(cost) / static_cast<T>(count);
}

template<typename T>
void ncf::Layer<T>::grad(const Stock<T>& in, Stock<T>& out, const std::function<T(const T&)>& div_cost) const {
	std::size_t prev_neurons = in.getLayer().getNeurons();
	out.createGrad(prev_neurons);

	Mat<T>& error = out.getError();
	Mat<T>& grad = out.getGrad(prev_neurons);
	const Mat<T>& prev_out = in.getConstOut();

	std::size_t count = error.getW() * error.getH();

	error.map(div_cost, error);
	error.mul(prev_out, grad, mcf::TRANSPOSE::SECOND);
	grad.map([&](const T& v) {
		return -v / static_cast<T>(count);
	}, grad);
}

template<typename T>
void ncf::Layer<T>::train(const Stock<T>& in, Stock<T>& out, T learning_rate) {
	std::size_t prev_neurons = in.getLayer().getNeurons();

	createCore(prev_neurons);
	optimizer::gd<T>(getCore(prev_neurons), out.getGrad(prev_neurons), learning_rate);
}
template<typename T>
void ncf::Layer<T>::train(const Stock<T>& in, Stock<T>& out, T learning_rate, ecl::Computer& video) {
	std::size_t prev_neurons = in.getLayer().getNeurons();

	createCore(prev_neurons, video);
	optimizer::gd<T>(getCore(prev_neurons), out.getGrad(prev_neurons), learning_rate, video);
}

template<typename T>
void ncf::Layer<T>::grad(const Stock<T>& in, Stock<T>& out, const std::string& div_cost, ecl::Computer& video) const {
	std::size_t prev_neurons = in.getLayer().getNeurons();

	out.createGrad(prev_neurons, video);

	Mat<T>& error = out.getError();
	Mat<T>& grad = out.getGrad(prev_neurons);
	const Mat<T>& prev_out = in.getConstOut();
	
	std::string count = std::to_string(static_cast<T>(error.getW() * error.getH()));

	error.map(div_cost, error, video);
	error.mul(prev_out, grad, video, mcf::TRANSPOSE::SECOND);
	grad.map("ret = -v / " + count + ";", grad, video);
}

// Stock
template<typename T>
ncf::Stock<T>::Stock(const Layer<T>& layer, std::size_t examples) : layer(layer){
    out = mcf::Mat<T>(layer.getNeurons(), examples);
    preout = mcf::Mat<T>(layer.getNeurons(), examples);
    error = mcf::Mat<T>(layer.getNeurons(), examples);
}

template<typename T>
void ncf::Stock<T>::send(ecl::Computer& video){
    for(auto& p : grad){
        if(p.second != nullptr) video << p.second;
    }
    video << preout;
    video << error;
    video << out;
}
template<typename T>
void ncf::Stock<T>::receive(ecl::Computer& video){
    for(auto& p : grad){
		bool dump = p.second != nullptr;
        if(p.second != nullptr) video >> p.second;
    }
    video >> preout;
    video >> error;
    video >> out;
}
template<typename T>
void ncf::Stock<T>::grab(ecl::Computer& video){
    for(auto& p : grad){
        if(p.second != nullptr) p.second.grab(video);
    }
    preout.grab(video);
    error.grab(video);
    out.grab(video);
}
template<typename T>
void ncf::Stock<T>::release(ecl::Computer& video){
    for(auto& p : grad){
        if(p.second != nullptr) p.second.release(video);
    }
    preout.release(video);
    error.release(video);
    out.release(video);
}

namespace ncf{
    template<typename T>
    Computer& operator<<(Computer& video, Stock<T>& other){
        other.send(video);
        return video;
    }
    template<typename T>
    Computer& operator>>(Computer& video, Stock<T>& other){
        other.receive(video);
        return video;
    }
}

template<typename T>
mcf::Mat<T>& ncf::Stock<T>::getPreout(){
    return preout;
}
template<typename T>
mcf::Mat<T>& ncf::Stock<T>::getOut(){
    return out;
}
template<typename T>
mcf::Mat<T>& ncf::Stock<T>::getError(){
    return error;
}
template<typename T>
mcf::Mat<T>& ncf::Stock<T>::getGrad(std::size_t prev_neurons){
    return grad.at(prev_neurons);
}

template<typename T>
const mcf::Mat<T>& ncf::Stock<T>::getConstPreout() const{
    return preout;
}
template<typename T>
const mcf::Mat<T>& ncf::Stock<T>::getConstOut() const{
    return out;
}
template<typename T>
const mcf::Mat<T>& ncf::Stock<T>::getConstError() const{
    return error;
}
template<typename T>
const ncf::Layer<T>& ncf::Stock<T>::getLayer() const{
    return layer;
}

template<typename T>
bool ncf::Stock<T>::checkGrad(std::size_t prev_neurons) const{
    if(grad.find(prev_neurons) == grad.end()) return false;
    return true;
}
template<typename T>
void ncf::Stock<T>::createGrad(std::size_t prev_neurons){
    if(!checkGrad(prev_neurons)){
        if(layer.getCoreGen() == nullptr)
            throw std::runtime_error("Stock [create grad]: layer's coregen method unsetted");

        Mat<T> new_grad(layer.getNeurons(), prev_neurons);
        layer.getCoreGen()(new_grad);
        grad.emplace(prev_neurons, std::move(new_grad));
    }
}
template<typename T>
void ncf::Stock<T>::createGrad(std::size_t prev_neurons, ecl::Computer& video) {
	if (!checkGrad(prev_neurons)) {
		Mat<T> new_grad(layer.getNeurons(), prev_neurons);
		video << new_grad;
		layer.getComputerCoreGen()(new_grad, video);
		grad.emplace(prev_neurons, std::move(new_grad));
	}
}

template<typename T>
void ncf::Stock<T>::releaseGrad(std::size_t prev_neurons){
    auto it = grad.find(prev_neurons);
    if(it != grad.end()){
        grad.erase(it);
    }
}

// High-level API

// Net
template<typename T>
ncf::Net<T>::Net() {}

template<typename T>
ncf::Net<T>::Net(const std::vector<std::size_t>& neurons){
    for(auto n : neurons)
        layers.push_back(std::make_pair(new Layer<T>(n), true));
}

template<typename T>
void ncf::Net<T>::send(Computer& video){
    for(auto& p : layers) p.first->send(video);
}
template<typename T>
void ncf::Net<T>::receive(Computer& video){
    for(auto& p : layers) p.first->receive(video);
}
template<typename T>
void ncf::Net<T>::grab(Computer& video){
    for(auto& p : layers) p.first->grab(video);
}
template<typename T>
void ncf::Net<T>::release(Computer& video){
    for(auto& p : layers) p.first->release(video);
}

namespace ncf{
    template<typename T>
    Computer& operator<<(Computer& video, Net<T>& net){
        net.send(video);
        return video;
    }
    template<typename T>
    Computer& operator>>(Computer& video, Net<T>& net){
        net.receive(video);
        return video;
    }
}

template<typename T>
void ncf::Net<T>::setActivations(const std::function<T(const T&)>& activation){
    for(auto& p : layers) p.first->setActivation(activation);
}
template<typename T>
void ncf::Net<T>::setDerivatives(const std::function<T(const T&)>& derivative){
    for(auto& p : layers) p.first->setDerivative(derivative);
}

template<typename T>
void ncf::Net<T>::setActivations(const std::string& activation){
    for(auto& p : layers) p.first->setActivation(activation);
}
template<typename T>
void ncf::Net<T>::setDerivatives(const std::string& derivative){
    for(auto& p : layers) p.first->setDerivative(derivative);
}

template<typename T>
void ncf::Net<T>::setCoreGens(const std::function<void(mcf::Mat<T>&)>& coregen){
    for(auto& p : layers) p.first->setCoreGen(coregen);
}
template<typename T>
void ncf::Net<T>::setCoreGens(const std::function<void(mcf::Mat<T>&, ecl::Computer&)>& coregen){
    for(auto& p : layers) p.first->setCoreGen(coregen);
}

template<typename T>
std::size_t ncf::Net<T>::getLayersCount() const{
    return layers.size();
}
template<typename T>
const ncf::Layer<T>* ncf::Net<T>::getConstLayer(std::size_t index) const{
    return layers.at(index).first;
}

template<typename T>
ncf::Layer<T>* ncf::Net<T>::getLayer(std::size_t index){
    return layers.at(index).first;
}

template<typename T>
ncf::Net<T>::~Net(){
    for(auto& p : layers){
        if(p.second == true) delete p.first;
    }
    layers.clear();
}


// StockPool
template<typename T>
ncf::StockPool<T>::StockPool() {}

template<typename T>
ncf::StockPool<T>::StockPool(const ncf::Net<T>& net, std::size_t examples){
    size_t count = net.getLayersCount();
    for(size_t i = 0; i < count; i++)
        stocks.push_back(std::make_pair(new Stock<T>(*net.getConstLayer(i), examples), true));
}


template<typename T>
void ncf::StockPool<T>::send(Computer& video){
    for(auto& p : stocks) p.first->send(video);
}
template<typename T>
void ncf::StockPool<T>::receive(Computer& video){
    for(auto& p : stocks) p.first->receive(video);
}
template<typename T>
void ncf::StockPool<T>::grab(Computer& video){
    for(auto& p : stocks) p.first->grab(video);
}
template<typename T>
void ncf::StockPool<T>::release(Computer& video){
    for(auto& p : stocks) p.first->release(video);
}

namespace ncf{
    template<typename T>
    Computer& operator<<(Computer& video, StockPool<T>& net){
        net.send(video);
        return video;
    }
    template<typename T>
    Computer& operator>>(Computer& video, StockPool<T>& net){
        net.receive(video);
        return video;
    }
}

template<typename T>
std::size_t ncf::StockPool<T>::getStocksCount() const{
    return stocks.size();
}
template<typename T>
const ncf::Stock<T>* ncf::StockPool<T>::getConstStock(std::size_t index) const{
    return stocks.at(index).first;
}

template<typename T>
ncf::Stock<T>* ncf::StockPool<T>::getStock(std::size_t index){
    return stocks.at(index).first;
}


template<typename T>
ncf::StockPool<T>::~StockPool(){
    for(auto& p : stocks){
        if(p.second == true) delete p.first;
    }
    stocks.clear();
}