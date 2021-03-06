#pragma once
#include <variant>
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
        void query(const Mat<T>& in, Mat<T>& out) const;
        void query(const Mat<T>& in, Mat<T>& out, Computer&) const;

        void query(const Mat<T>& in, Mat<T>& preout, Mat<T>& out, const Layer<T>& prev);
        void query(const Mat<T>& in, Mat<T>& preout, Mat<T>& out, const Layer<T>& prev, Computer&);

        void error(const Mat<T>& answer, const Mat<T>& out, Mat<T>& error) const;
        void error(const Mat<T>& answer, const Mat<T>& out, Mat<T>& error, Computer&) const;

        void error(const Mat<T>& next_error, Mat<T>& preout, Mat<T>& error, const Layer<T>& next) const;
        void error(const Mat<T>& next_error, Mat<T>& preout, Mat<T>& error, const Layer<T>& next, Computer&) const;

        T cost(const Mat<T>& error, const std::function<T(const T&)>& cost) const;

        void grad(Mat<T>& error, const Mat<T>& prev_out, Mat<T>& grad, const std::function<T(const T&)>& div_cost) const;
        void grad(Mat<T>& error, const Mat<T>& prev_out, Mat<T>& grad, const std::string& div_cost, Computer&) const;

        void train(Mat<T>& grad, const Layer<T>& prev, const T& learning_rate);
        void train(Mat<T>& grad, const Layer<T>& prev, const T& learning_rate, Computer& video);

        // High-level methods
        void query(const Mat<T>& in, Stock<T>& stock) const;
        void query(const Mat<T>& in, Stock<T>& stock, Computer&) const;

        void query(const Stock<T>& prev_stock, Stock<T>& stock);
        void query(const Stock<T>& prev_stock, Stock<T>& stock, Computer&);

        void error(const Mat<T>& in, Stock<T>& stock) const;
        void error(const Mat<T>& in, Stock<T>& stock, Computer&) const;

        void error(const Stock<T>& next_stock, Stock<T>& stock) const;
        void error(const Stock<T>& next_stock, Stock<T>& stock, Computer&) const;

        T cost(const Stock<T>& stock, const std::function<T(const T&)>& cost) const;

		void grad(const Stock<T>& prev_stock, Stock<T>& stock, const std::function<T(const T&)>& div_cost) const;
		void grad(const Stock<T>& prev_stock, Stock<T>& stock, const std::string& div_cost, Computer&) const;

		void train(const Stock<T>& prev_stock, Stock<T>& stock, const T& learning_rate);
		void train(const Stock<T>& prev_stock, Stock<T>& stock, const T& learning_rate, Computer&);
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
		template<typename U>
		friend std::ostream& operator<<(std::ostream&, Stock<U>&);

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
	struct FitFrame {
		const Mat<T>& data;
		const Mat<T>& answer;
		StockPool<T>& pool;
		std::function<T(const T&)> cost;
		std::variant<std::function<T(const T&)>, std::string> div_cost;
	};

    template<typename T>
    class Net{
    private:
        std::vector<std::pair<Layer<T>*, bool>> layers;

        void checkStockPool(const StockPool<T>&, const std::string&) const;
    public:
        Net();
        explicit Net(const std::vector<std::size_t>&);
		explicit Net(const std::vector<Layer<T>*>&);

        void send(Computer&);
        void receive(Computer&);
        void grab(Computer&);
        void release(Computer&);

        template<typename U>
        friend Computer& operator<<(Computer&, Net<U>&);
        template<typename U>
        friend Computer& operator>>(Computer&, Net<U>&);

		void push_back(Layer<T>*);
		Layer<T>* pop_back();

        // total setters
        void setActivations(const std::function<T(const T&)>&);
        void setDerivatives(const std::function<T(const T&)>&);

        void setActivations(const std::string&);
        void setDerivatives(const std::string&);

        void setCoreGens(const std::function<void(Mat<T>&)>&);
		void setCoreGens(const std::function<void(Mat<T>&, Computer&)>&);

        // partial setters
        void setActivations(const std::vector<std::size_t>&, const std::function<T(const T&)>&);
        void setDerivatives(const std::vector<std::size_t>&, const std::function<T(const T&)>&);

        void setActivations(const std::vector<std::size_t>&, const std::string&);
        void setDerivatives(const std::vector<std::size_t>&, const std::string&);

        void setCoreGens(const std::vector<std::size_t>&, const std::function<void(Mat<T>&)>&);
		void setCoreGens(const std::vector<std::size_t>&, const std::function<void(Mat<T>&, Computer&)>&);

        std::size_t getLayersCount() const;
        const Layer<T>& getConstLayer(std::size_t) const;
        Layer<T>& getLayer(std::size_t);

        // Low-level methods
        void query(const Mat<T>& in, StockPool<T>& pool);
        void query(const Mat<T>& in, StockPool<T>& pool, Computer&);

        void error(const Mat<T>& answer, StockPool<T>& pool);
        void error(const Mat<T>& answer, StockPool<T>& pool, Computer&);

        T cost(const StockPool<T>& pool, const std::function<T(const T&)>& cost) const;

        void grad(StockPool<T>& pool, const std::function<T(const T&)>& div_cost);
        void grad(StockPool<T>& pool, const std::string& div_cost, Computer&);

        void train(StockPool<T>& pool, const T& learning_rate);
        void train(StockPool<T>& pool, const T& learning_rate, Computer&);

		// High-level methods
		T fit(const FitFrame<T>& frame, const T& learning_rate, std::size_t max_iterations, const T& min_error);
		T fit(const FitFrame<T>& frame, const T& learning_rate, std::size_t max_iterations, const T& min_error, Computer&);

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
        friend Computer& operator<<(Computer&, StockPool<U>&);
        template<typename U>
        friend Computer& operator>>(Computer&, StockPool<U>&);
		template<typename U>
		friend std::ostream& operator<<(std::ostream&, StockPool<U>&);

		void push_back(Stock<T>*);
		Stock<T>* pop_back();

        std::size_t getStocksCount() const;
        const Stock<T>& getConstStock(std::size_t) const;
        Stock<T>& getStock(std::size_t);
		Stock<T>& getLastStock();

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
        void gd(Mat<T>& X, Mat<T>& grad, const T& learning_rate){
            grad.mul(learning_rate, grad);
            X.sub(grad, X);
        }
        template<typename T>
        void gd(Mat<T>& X, Mat<T>& grad, const T& learning_rate, ecl::Computer& video){
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
void ncf::Layer<T>::train(mcf::Mat<T>& grad, const Layer<T>& prev, const T& learning_rate){
    createCore(prev.neurons);
    optimizer::gd<T>(getCore(prev.neurons), grad, learning_rate);
}
template<typename T>
void ncf::Layer<T>::train(mcf::Mat<T>& grad, const Layer<T>& prev, const T& learning_rate, ecl::Computer& video){
    createCore(prev.neurons, video);

    optimizer::gd<T>(getCore(prev.neurons), grad, learning_rate, video);
}

// High-level methods
template<typename T>
void ncf::Layer<T>::query(const mcf::Mat<T>& in, Stock<T>& stock) const{
	query(in, stock.getOut());
}
template<typename T>
void ncf::Layer<T>::query(const mcf::Mat<T>& in, Stock<T>& stock, ecl::Computer& video) const{
	query(in, stock.getOut(), video);
}

template<typename T>
void ncf::Layer<T>::query(const Stock<T>& prev_stock, Stock<T>& stock){
	query(prev_stock.getConstOut(), stock.getPreout(), stock.getOut(), prev_stock.getLayer());
}
template<typename T>
void ncf::Layer<T>::query(const Stock<T>& prev_stock, Stock<T>& stock, ecl::Computer& video){
	query(prev_stock.getConstOut(), stock.getPreout(), stock.getOut(), prev_stock.getLayer(), video);
}

template<typename T>
void ncf::Layer<T>::error(const mcf::Mat<T>& answer, Stock<T>& stock) const{
	error(answer, stock.getConstOut(), stock.getError());
}
template<typename T>
void ncf::Layer<T>::error(const mcf::Mat<T>& answer, Stock<T>& stock, ecl::Computer& video) const{
	error(answer, stock.getConstOut(), stock.getError(), video);
}

template<typename T>
void ncf::Layer<T>::error(const Stock<T>& next_stock, Stock<T>& stock) const{
	error(next_stock.getConstError(), stock.getPreout(), stock.getError(), next_stock.getLayer());
}
template<typename T>
void ncf::Layer<T>::error(const Stock<T>& next_stock, Stock<T>& stock, ecl::Computer& video) const{
	error(next_stock.getConstError(), stock.getPreout(), stock.getError(), next_stock.getLayer(), video);
}

template<typename T>
T ncf::Layer<T>::cost(const Stock<T>& stock, const std::function<T(const T&)>& cost) const{
	return this->cost(stock.getConstError(), cost);
}

template<typename T>
void ncf::Layer<T>::grad(const Stock<T>& prev_stock, Stock<T>& stock, const std::function<T(const T&)>& div_cost) const {
	std::size_t prev_neurons = prev_stock.getLayer().getNeurons();
	stock.createGrad(prev_neurons);

	grad(stock.getError(), prev_stock.getConstOut(), stock.getGrad(prev_neurons), div_cost);
}
template<typename T>
void ncf::Layer<T>::grad(const Stock<T>& prev_stock, Stock<T>& stock, const std::string& div_cost, ecl::Computer& video) const {
	std::size_t prev_neurons = prev_stock.getLayer().getNeurons();
	stock.createGrad(prev_neurons, video);

	grad(stock.getError(), prev_stock.getConstOut(), stock.getGrad(prev_neurons), div_cost, video);
}

template<typename T>
void ncf::Layer<T>::train(const Stock<T>& prev_stock, Stock<T>& stock, const T& learning_rate) {
	std::size_t prev_neurons = prev_stock.getLayer().getNeurons();

	optimizer::gd<T>(getCore(prev_neurons), stock.getGrad(prev_neurons), learning_rate);
}
template<typename T>
void ncf::Layer<T>::train(const Stock<T>& prev_stock, Stock<T>& stock, const T& learning_rate, ecl::Computer& video) {
	std::size_t prev_neurons = prev_stock.getLayer().getNeurons();

	optimizer::gd<T>(getCore(prev_neurons), stock.getGrad(prev_neurons), learning_rate, video);
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
	template<typename T>
	std::ostream& operator<<(std::ostream& s, Stock<T>& other) {
		s << other.getConstOut();
		return s;
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
void ncf::Net<T>::checkStockPool(const ncf::StockPool<T>& pool, const std::string& where) const{
    if(pool.getStocksCount() != layers.size())
        throw std::runtime_error("Net [" + where + "]: invalid pool");
}

template<typename T>
ncf::Net<T>::Net() {}

template<typename T>
ncf::Net<T>::Net(const std::vector<std::size_t>& neurons){
    for(auto n : neurons)
        layers.push_back(std::make_pair(new Layer<T>(n), true));
}
template<typename T>
ncf::Net<T>::Net(const std::vector<Layer<T>*>& layers) {
	for (auto* l : layers)
		this->layers.push_back(std::make_pair(l, false));
}

template<typename T>
void ncf::Net<T>::send(ecl::Computer& video){
    for(auto& p : layers) p.first->send(video);
}
template<typename T>
void ncf::Net<T>::receive(ecl::Computer& video){
    for(auto& p : layers) p.first->receive(video);
}
template<typename T>
void ncf::Net<T>::grab(ecl::Computer& video){
    for(auto& p : layers) p.first->grab(video);
}
template<typename T>
void ncf::Net<T>::release(ecl::Computer& video){
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
void ncf::Net<T>::push_back(Layer<T>* layer) {
	layers.push_back(std::make_pair(layer, false));
}
template<typename T>
ncf::Layer<T>* ncf::Net<T>::pop_back() {
	Layer<T>* result = layers.back().first;
	layers.pop_back();
	return result;
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
void ncf::Net<T>::setActivations(const std::vector<std::size_t>& indexes, const std::function<T(const T&)>& activation){
    for(auto n : indexes) layers.at(n).first->setActivation(activation);
}
template<typename T>
void ncf::Net<T>::setDerivatives(const std::vector<std::size_t>& indexes, const std::function<T(const T&)>& derivative){
    for(auto n : indexes) layers.at(n).first->setDerivative(derivative);
}

template<typename T>
void ncf::Net<T>::setActivations(const std::vector<std::size_t>& indexes, const std::string& activation){
    for(auto n : indexes) layers.at(n).first->setActivation(activation);
}
template<typename T>
void ncf::Net<T>::setDerivatives(const std::vector<std::size_t>& indexes, const std::string& derivative){
    for(auto n : indexes) layers.at(n).first->setDerivative(derivative);
}

template<typename T>
void ncf::Net<T>::setCoreGens(const std::vector<std::size_t>& indexes, const std::function<void(mcf::Mat<T>&)>& coregen){
    for(auto n : indexes) layers.at(n).first->setCoreGen(coregen);
}
template<typename T>
void ncf::Net<T>::setCoreGens(const std::vector<std::size_t>& indexes, const std::function<void(mcf::Mat<T>&, ecl::Computer&)>& coregen){
    for(auto n : indexes) layers.at(n).first->setCoreGen(coregen);
}

template<typename T>
std::size_t ncf::Net<T>::getLayersCount() const{
    return layers.size();
}
template<typename T>
const ncf::Layer<T>& ncf::Net<T>::getConstLayer(std::size_t index) const{
    return *layers.at(index).first;
}

template<typename T>
ncf::Layer<T>& ncf::Net<T>::getLayer(std::size_t index){
    return *layers.at(index).first;
}

template<typename T>
void ncf::Net<T>::query(const mcf::Mat<T>& in, StockPool<T>& pool){
    checkStockPool(pool, "query");

    layers.at(0).first->query(in, pool.getStock(0));

    size_t count = pool.getStocksCount();
    for(size_t i = 1; i < count; i++)
        layers.at(i).first->query(pool.getStock(i - 1), pool.getStock(i));
}
template<typename T>
void ncf::Net<T>::query(const mcf::Mat<T>& in, StockPool<T>& pool, ecl::Computer& video){
    checkStockPool(pool, "query");

    layers.at(0).first->query(in, pool.getStock(0), video);

    size_t count = pool.getStocksCount();
    for(size_t i = 1; i < count; i++)
        layers.at(i).first->query(pool.getStock(i - 1), pool.getStock(i), video);
}

template<typename T>
void ncf::Net<T>::error(const mcf::Mat<T>& answer, StockPool<T>& pool){
    checkStockPool(pool, "error");

    size_t count = pool.getStocksCount();
    size_t last = count - 1;

    layers.at(last).first->error(answer, pool.getStock(last));

    for(int i = last - 1; i >= 1; i--)
        layers.at(i).first->error(pool.getConstStock(i + 1), pool.getStock(i));
}
template<typename T>
void ncf::Net<T>::error(const mcf::Mat<T>& answer, StockPool<T>& pool, ecl::Computer& video){
    checkStockPool(pool, "error");

    size_t count = pool.getStocksCount();
    size_t last = count - 1;

    layers.at(last).first->error(answer, pool.getStock(last), video);

    for(int i = last - 1; i >= 1; i--)
        layers.at(i).first->error(pool.getConstStock(i + 1), pool.getStock(i), video);
}

template<typename T>
T ncf::Net<T>::cost(const StockPool<T>& pool, const std::function<T(const T&)>& cost) const{
    checkStockPool(pool, "cost");

    size_t count = pool.getStocksCount();
    size_t last = count - 1;

    return layers.at(last).first->cost(pool.getConstStock(last), cost);
}

template<typename T>
void ncf::Net<T>::grad(StockPool<T>& pool, const std::function<T(const T&)>& div_cost){
    checkStockPool(pool, "grad");

    size_t count = pool.getStocksCount();
    
    for(size_t i = 1; i < count; i++)
        layers.at(i).first->grad(pool.getConstStock(i - 1), pool.getStock(i), div_cost);
}
template<typename T>
void ncf::Net<T>::grad(StockPool<T>& pool, const std::string& div_cost, ecl::Computer& video){
    checkStockPool(pool, "grad");

    size_t count = pool.getStocksCount();
    
    for(size_t i = 1; i < count; i++)
        layers.at(i).first->grad(pool.getConstStock(i - 1), pool.getStock(i), div_cost, video);
}

template<typename T>
void ncf::Net<T>::train(StockPool<T>& pool, const T& learning_rate){
    checkStockPool(pool, "train");

    size_t count = pool.getStocksCount();
    
    for(size_t i = 1; i < count; i++)
        layers.at(i).first->train(pool.getConstStock(i - 1), pool.getStock(i), learning_rate);
}
template<typename T>
void ncf::Net<T>::train(StockPool<T>& pool, const T& learning_rate, ecl::Computer& video){
    checkStockPool(pool, "train");

    size_t count = pool.getStocksCount();
    
    for(size_t i = 1; i < count; i++)
        layers.at(i).first->train(pool.getConstStock(i - 1), pool.getStock(i), learning_rate, video);
}

template<typename T>
T ncf::Net<T>::fit(const FitFrame<T>& frame, const T& learning_rate, std::size_t max_iterations, const T& min_error) {
	const mcf::Mat<T>& data = frame.data;
	const mcf::Mat<T>& answer = frame.answer;
	StockPool<T>& pool = frame.pool;
	const std::function<T(const T&)>& cost = frame.cost;
	const std::function<T(const T&)>& div_cost = std::get<0>(frame.div_cost);

	T e = 1;
	for (size_t i = 0; i < max_iterations; i++) {
		query(data, pool);
		error(answer, pool);

		e = this->cost(pool, cost);
		if (e < min_error) break;

		grad(pool, div_cost);
		train(pool, learning_rate);
	}

	return e;
}
template<typename T>
T ncf::Net<T>::fit(const FitFrame<T>& frame, const T& learning_rate, std::size_t max_iterations, const T& min_error, ecl::Computer& video) {
	const mcf::Mat<T>& data = frame.data;
	const mcf::Mat<T>& answer = frame.answer;
	StockPool<T>& pool = frame.pool;
	const std::function<T(const T&)>& cost = frame.cost;
	const std::string& div_cost = std::get<1>(frame.div_cost);

	T e = 1;
	for (size_t i = 0; i < max_iterations; i++) {
		query(data, pool, video);
		error(answer, pool, video);

		video >> pool.getStock(pool.getStocksCount() - 1).getError();
		e = this->cost(pool, cost);
		if (e < min_error) break;

		grad(pool, div_cost, video);
		train(pool, learning_rate, video);
	}

	return e;
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
        stocks.push_back(std::make_pair(new Stock<T>(net.getConstLayer(i), examples), true));
}


template<typename T>
void ncf::StockPool<T>::send(ecl::Computer& video){
    for(auto& p : stocks) p.first->send(video);
}
template<typename T>
void ncf::StockPool<T>::receive(ecl::Computer& video){
    for(auto& p : stocks) p.first->receive(video);
}
template<typename T>
void ncf::StockPool<T>::grab(ecl::Computer& video){
    for(auto& p : stocks) p.first->grab(video);
}
template<typename T>
void ncf::StockPool<T>::release(ecl::Computer& video){
    for(auto& p : stocks) p.first->release(video);
}

namespace ncf{
    template<typename T>
    Computer& operator<<(Computer& video, StockPool<T>& pool){
        pool.send(video);
        return video;
    }
    template<typename T>
    Computer& operator>>(Computer& video, StockPool<T>& pool){
        pool.receive(video);
        return video;
    }
	template<typename T>
	std::ostream& operator<<(std::ostream& s, StockPool<T>& pool) {
		s << pool.getStock(pool.getStocksCount() - 1);
		return s;
	}
}

template<typename T>
void ncf::StockPool<T>::push_back(Stock<T>* stock) {
	stocks.push_back(std::make_pair(stock, false));
}
template<typename T>
ncf::Stock<T>* ncf::StockPool<T>::pop_back() {
	Layer<T>* result = stocks.back().first;
	stocks.pop_back();
	return result;
}

template<typename T>
std::size_t ncf::StockPool<T>::getStocksCount() const{
    return stocks.size();
}
template<typename T>
const ncf::Stock<T>& ncf::StockPool<T>::getConstStock(std::size_t index) const{
    return *stocks.at(index).first;
}

template<typename T>
ncf::Stock<T>& ncf::StockPool<T>::getStock(std::size_t index){
    return *stocks.at(index).first;
}
template<typename T>
ncf::Stock<T>& ncf::StockPool<T>::getLastStock() {
	return *stocks.back().first;
}


template<typename T>
ncf::StockPool<T>::~StockPool(){
    for(auto& p : stocks){
        if(p.second == true) delete p.first;
    }
    stocks.clear();
}